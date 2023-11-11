import torch
import torch.nn as nn
from loss import ModelLoss
import torch.optim as optim

"""
All TODO s on this page are fixes done when translating tensorflow code to pytorch. If training results aren't
what we were expecting, these may be places to look.
"""
class VFE_Layer(nn.Module):
    def __init__(self, c_in, c_out, device):
        """
            A VFE layer class
            Args:
                c_in: int, the dimensions of the input
                c_out : int, the dimension of the output after VFE, must be even
        """
        super(VFE_Layer, self).__init__()
        self.device = device
        self.in_units = c_in
        self.out_units = c_out // 2
        self.fcn = nn.Linear(self.in_units, self.out_units).to(device)
        self.bn = nn.BatchNorm2d(self.out_units).to(device)

    def forward(self, input, mask):
        """
        Forward method of the class
        Args:
            input : Tensor (4D tensor in our case), [Batch_size, max_num_voxels, max_num_pts, out_dim]
            (out_dim = 7, at the beginning of the network)
        Returns:
            output : Tensor with the same shape as input, except the last dim which is c_out
        """
        fcn_out = torch.relu(self.fcn(input.to(self.device)))
        fcn_out = fcn_out.permute((0,3,2,1)) # TODO: This has to be done to keep dims right, find alternative/see if this is causing bugs
        fcn_out = self.bn(fcn_out)
        fcn_out = fcn_out.permute((0,3,2,1))
        max_pool = fcn_out.max(dim=2, keepdim=True)[0]
        tiled_max_pool = max_pool.repeat(1, 1, input.size(2), 1)
        output = torch.cat([fcn_out, tiled_max_pool], dim=-1)
        mask = mask.repeat(1, 1, 1, 2 * self.out_units)
        ret = output * mask.float()
        return ret


class VFE_Block(nn.Module):
    def __init__(self, vfe_out_dims, final_dim, sparse_shape, device):
        """
            VFE_block class, made of VFE layers
            
            Args:
            vfe_out_dims : n-integer list made of the output dimensions of VFEs, each dimension must be even
            final_dim : int32, dimension of the last Dense layer after VFEs
            sparse_shape : 3-list, int32, dimensions of the sparse voxels space // ex : [10, 400,352] 
        """
        super(VFE_Block, self).__init__()
        self.device = device
        self.vfe_out_dims = vfe_out_dims
        self.vfe_in_dims = [7] + vfe_out_dims[:-1]
        self.final_dim = final_dim
        self.sparse_shape = sparse_shape
        self.VFEs = [VFE_Layer(in_dim, out_dim, device) for in_dim, out_dim in zip(self.vfe_in_dims, self.vfe_out_dims)]
        self.final_fcn = nn.Linear(vfe_out_dims[-1], final_dim)

    def forward(self, input, voxel_coor_buffer, shape, training=False):
        """
        Forward Method
        Args:
            input : 4D tensor, of type float32, [batch_size, K, T, 7]
            voxel_coor_buffer : 2D tensor , int32 of dimension [batch_size, 4]
            training : (optional), boolean 
        Returns:
            output : 5-D tensor, [batch_size, channels, Depth, Height, Width]
        """
        mask = (input.max(dim=-1, keepdim=True)[0] != 0)
        vfe_out = input.to(self.device)
        for vfe in self.VFEs:
            vfe_out = vfe(vfe_out, mask).to(self.device)
        output = self.final_fcn(vfe_out).max(dim=2)[0] # [batch_size, max_num_voxels, final_dim]
        sparse_output = torch.zeros(shape, dtype=output.dtype, device=output.device) # [2, 10, 200, 240, 128]
        # TODO: See if this effectively replaces tf.scatter as done here: https://github.com/gkadusumilli/Voxelnet/blob/d6865c8cc53bbc3150d8e1249afd65e4eb231142/model.py#L83
        sparse_output[:,voxel_coor_buffer[:,:,0],voxel_coor_buffer[:,:,1],voxel_coor_buffer[:,:,2],:] = output # [batch_size, Depth, Height, Width, channels]
        return sparse_output.permute(0, 4, 1, 2, 3) #[batch_size, channels, Depth, Height, Width]


class ConvMiddleLayer(nn.Module):

    def __init__(self, out_shape, device):
        """
            Convolutional Middle Layer class
            Args:
            out_shape : 4-list, int32, dimensions of the output (batch_size, new_chnnles, height, widht)
        """
        super(ConvMiddleLayer, self).__init__()
        self.device = device
        self.out_shape = out_shape
        self.conv1 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1,1,1))
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0,1,1))
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1,1,1))
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)

    def forward(self, input):
        """
        Forward Method
        Args:
            input : 5D Tensor, float32, shape=[batch_size, channels(128), Depth(10), Height(400), Width(352)]
        returns:
            4D tensor, float32, shape=(batch_size, new_chnnles, height, widht)
        """
        out = torch.relu(self.bn1(self.conv1(input)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = torch.relu(self.bn3(self.conv3(out)))
        return out.reshape(*self.out_shape)


class RPN(nn.Module):
    def __init__(self, num_anchors_per_cell, device):
        super(RPN, self).__init__()
        self.device = device
        self.num_anchors_per_cell = num_anchors_per_cell
        
        # Batch Normalization
        BN = nn.BatchNorm2d
        
        # Block 1
        self.conv1_block1, self.bn1_block1 = self.conv_layer(128, 128, (3, 3), (2, 2), 1), BN(128)
        self.conv2_block1, self.bn2_block1 = self.conv_layer(128, 128, (3, 3), (1, 1), 1), BN(128)
        self.conv3_block1, self.bn3_block1 = self.conv_layer(128, 128, (3, 3), (1, 1), 1), BN(128)
        self.conv4_block1, self.bn4_block1 = self.conv_layer(128, 128, (3, 3), (1, 1), 1), BN(128)
        
        # Block 2
        self.conv1_block2, self.bn1_block2 = self.conv_layer(128, 128, (3, 3), (2, 2), 1), BN(128)
        self.conv2_block2, self.bn2_block2 = self.conv_layer(128, 128, (3, 3), (1, 1), 1), BN(128)
        self.conv3_block2, self.bn3_block2 = self.conv_layer(128, 128, (3, 3), (1, 1), 1), BN(128)
        self.conv4_block2, self.bn4_block2 = self.conv_layer(128, 128, (3, 3), (1, 1), 1), BN(128)
        self.conv5_block2, self.bn5_block2 = self.conv_layer(128, 128, (3, 3), (1, 1), 1), BN(128)
        self.conv6_block2, self.bn6_block2 = self.conv_layer(128, 128, (3, 3), (1, 1), 1), BN(128)
        # Block 3
        self.conv1_block3, self.bn1_block3 = self.conv_layer(128, 256, (3, 3), (2, 2), 1), BN(256)
        self.conv2_block3, self.bn2_block3 = self.conv_layer(256, 256, (3, 3), (1, 1), 1), BN(256)
        self.conv3_block3, self.bn3_block3 = self.conv_layer(256, 256, (3, 3), (1, 1), 1), BN(256)
        self.conv4_block3, self.bn4_block3 = self.conv_layer(256, 256, (3, 3), (1, 1), 1), BN(256)
        self.conv5_block3, self.bn5_block3 = self.conv_layer(256, 256, (3, 3), (1, 1), 1), BN(256)
        self.conv6_block3, self.bn6_block3 = self.conv_layer(256, 256, (3, 3), (1, 1), 1), BN(256)
        
        # Deconvolutions
        self.deconv_1, self.deconv_bn1 = self.deconv_layer(128, 256, (3, 3), (1, 1), 1), BN(256)
        self.deconv_2, self.deconv_bn2 = self.deconv_layer(128, 256, (2, 2), (2, 2), 0), BN(256)
        self.deconv_3, self.deconv_bn3 = self.deconv_layer(256, 256, (4, 4), (4, 4), 0), BN(256)
        
        # Probability and regression maps
        self.prob_map_conv = self.conv_layer(768, self.num_anchors_per_cell, (1, 1), (1, 1), 0)
        self.reg_map_conv = self.conv_layer(768, 7 * self.num_anchors_per_cell, (1, 1), (1, 1), 0)
        
    def conv_layer(self, in_channels, out_channels, kernel_size, stride_size, padding):
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride_size, padding=padding)
    
    def deconv_layer(self, in_channels, out_channels, kernel_size, stride_size, padding):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride_size, padding=padding)
    
    def block_conv_op(self, block_id, input):
        i = 1
        out = input
        while True:
            try:
                c = getattr(self, "conv{}_block{}".format(i, block_id))
                b = getattr(self, "bn{}_block{}".format(i, block_id))
            except:
                break
            out = torch.relu(b(c(out)))
            i += 1
        return out
    
    def deconv_op(self, i, input):
        out = input
        c = getattr(self, "deconv_{}".format(i))
        b = getattr(self, "deconv_bn{}".format(i))
        out = torch.relu(b(c(out)))
        return out
    
    def forward(self, input):
        input_shape = input.shape
        assert len(input_shape) == 4 and input_shape[-1] % 8 == 0 and input_shape[-2] % 8 == 0, "The input must be of shape [Batch_size, channels, map_height, map_width] with map_height and map_width multiple of 8, got {}".format(input_shape)
        
        output = self.block_conv_op(1, input)
        deconv1 = self.deconv_op(1, output)
        
        output = self.block_conv_op(2, output)
        deconv2 = self.deconv_op(2, output)
        
        output = self.block_conv_op(3, output)
        deconv3 = self.deconv_op(3, output)
        output = torch.cat([deconv3, deconv2, deconv1], dim=1)
        prob_map = self.prob_map_conv(output)
        reg_map = self.reg_map_conv(output)
        
        prob_map = torch.sigmoid(prob_map)
        return prob_map.permute(0, 2, 3, 1), reg_map.permute(0, 2, 3, 1) # TODO: Determine if this permute is correct.

class Model(nn.Module):
    def __init__(self, cfg, params, device):
        super(Model, self).__init__()
        self.device = device
        self.params = params
        self.cfg = cfg
        self.vfe_block = VFE_Block(cfg.VFE_OUT_DIMS, cfg.VFE_FINAl_OUT_DIM, cfg.GRID_SIZE, device)
        self.convMiddle = ConvMiddleLayer((params["batch_size"], cfg.VFE_FINAl_OUT_DIM, *cfg.GRID_SIZE[1:]), device)
        self.rpn = RPN(cfg.NUM_ANCHORS_PER_CELL, device)
        self.loss_object = ModelLoss(self.params)
        boundaries = [80, 120]
        self.lr_cst = params["learning_rate"]
        self.optimizer = optim.Adam(self.parameters(), lr=params["learning_rate"])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=boundaries, gamma=0.1)
        
            

    def forward(self, feature_buffer, coordinate_buffer):
        shape = [self.params["batch_size"], *self.vfe_block.sparse_shape, self.vfe_block.final_dim] 
        output = self.vfe_block(feature_buffer, coordinate_buffer, shape, True)  # 'training' is set to True
        output = self.convMiddle(output)
        prob_map, reg_map = self.rpn(output)
        return prob_map, reg_map

    def train_step(self, feature_buffer, coordinate_buffer, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum):
        p_map, r_map = self(feature_buffer, coordinate_buffer)
        loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = self.loss_object(r_map, p_map, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)

        return loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss

    def validate_step(self, feature_buffer, coordinate_buffer, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum):
        p_map, r_map = self(feature_buffer, coordinate_buffer)
        loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = self.loss_object(r_map, p_map, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)
        return loss.item(), reg_loss.item(), cls_loss.item(), cls_pos_loss.item(), cls_neg_loss.item()

    def predict_step(self, feature_buffer, coordinate_buffer):
        p_map, r_map = self(feature_buffer, coordinate_buffer)
        return p_map, r_map
