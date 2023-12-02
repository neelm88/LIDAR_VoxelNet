import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
All TODO s on this page are fixes done when translating tensorflow code to pytorch. If training results aren't
what we were expecting, these may be places to look.
"""
class VFE_Layer(nn.Module):
    def __init__(self, c_in, c_out, requires_grad=True):
        """
            A VFE layer class
            Args:
                c_in: int, the dimensions of the input
                c_out : int, the dimension of the output after VFE, must be even
        """
        super(VFE_Layer, self).__init__()
        self.out_units = c_out // 2
        self.fcn = nn.Sequential(
            nn.Linear(c_in, self.out_units), 
            nn.ReLU()
        ).to(device)
        self.bn = nn.BatchNorm2d(self.out_units).to(device)
        self.bn.train(requires_grad)
        for p in self.parameters():
            p.requires_grad = requires_grad
        
    def forward(self, input, mask):
        """
        Forward method of the class
        Args:
            input : Tensor (4D tensor in our case), [Batch_size, max_num_voxels, max_num_pts, out_dim]
            (out_dim = 7, at the beginning of the network)
        Returns:
            output : Tensor with the same shape as input, except the last dim which is c_out
        """
        fcn_out = self.fcn(input).permute((0,3,2,1)) # TODO: This has to be done to keep dims right, find alternative/see if this is causing bugs
        fcn_out = self.bn(fcn_out).permute((0,3,2,1))

        max_pool = fcn_out.max(dim=2, keepdim=True)[0]
        tiled_max_pool = max_pool.repeat(1, 1, input.size(2), 1)
        output = torch.cat([fcn_out, tiled_max_pool], dim=-1)
        mask = mask.repeat(1, 1, 1, 2 * self.out_units)
        ret = output * mask.float()
        return ret


class VFE_Block(nn.Module):
    def __init__(self, vfe_out_dims, final_dim, sparse_shape, requires_grad=True):
        """
            VFE_block class, made of VFE layers
            
            Args:
            vfe_out_dims : n-integer list made of the output dimensions of VFEs, each dimension must be even
            final_dim : int32, dimension of the last Dense layer after VFEs
            sparse_shape : 3-list, int32, dimensions of the sparse voxels space // ex : [10, 400,352] 
        """
        super(VFE_Block, self).__init__()
        self.vfe_out_dims   = vfe_out_dims
        self.vfe_in_dims    = [7] + vfe_out_dims[:-1]
        self.final_dim      = final_dim
        self.sparse_shape   = sparse_shape
        self.VFEs           = [VFE_Layer(in_dim, out_dim, requires_grad=requires_grad).to(device) for in_dim, out_dim in zip(self.vfe_in_dims, self.vfe_out_dims)]
        self.final_fcn      = nn.Linear(vfe_out_dims[-1], final_dim).to(device)

        for p in self.parameters():
            p.requires_grad = requires_grad

        shape               = [1, *self.sparse_shape, self.final_dim] # [1, 10, 200, 240, 128]
        self.sparse_output  = torch.zeros(shape, dtype=torch.float32, requires_grad=requires_grad).to(device)

    def forward(self, input, voxel_coor_buffer):
        """
        Forward Method
        Args:
            input : 4D tensor, of type float32, [batch_size, K, T, 7]
            voxel_coor_buffer : 2D tensor , int32 of dimension [batch_size, 4]
            training : (optional), boolean 
        Returns:
            output : 5-D tensor, [batch_size, channels, Depth, Height, Width]
        """
        mask = (input.max(dim=-1, keepdim=True)[0] != 0).detach()
        vfe_out = input
        for vfe in self.VFEs:
            vfe_out = vfe(vfe_out, mask)
        output = self.final_fcn(vfe_out).max(dim=2)[0] # [batch_size, max_num_voxels, final_dim]
        
        # TODO: See if this effectively replaces tf.scatter as done here: https://github.com/gkadusumilli/Voxelnet/blob/d6865c8cc53bbc3150d8e1249afd65e4eb231142/model.py#L83
        self.sparse_output.zero_()
        self.sparse_output[:, voxel_coor_buffer[:,:,0], voxel_coor_buffer[:,:,1], voxel_coor_buffer[:,:,2], :] = output # [batch_size, Depth, Height, Width, channels]
        return self.sparse_output.permute(0, 4, 1, 2, 3) #[batch_size, channels, Depth, Height, Width]


class ConvMiddleLayer(nn.Module):

    def __init__(self, out_shape, requires_grad=True):
        """
            Convolutional Middle Layer class
            Args:
            out_shape : 4-list, int32, dimensions of the output (batch_size, new_chnnles, height, width)
        """
        super(ConvMiddleLayer, self).__init__()
        self.out_shape  = out_shape
        self.model      = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1,1,1)), nn.BatchNorm3d(64).train(requires_grad), nn.ReLU(),
            nn.Conv3d(64, 64,  kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0,1,1)), nn.BatchNorm3d(64).train(requires_grad), nn.ReLU(),
            nn.Conv3d(64, 64,  kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1,1,1)), nn.BatchNorm3d(64).train(requires_grad), nn.ReLU()
        ).to(device)
        for p in self.parameters():
            p.requires_grad = requires_grad
            

    def forward(self, input):
        """
        Forward Method
        Args:
            input : 5D Tensor, float32, shape=[batch_size, channels(128), Depth(10), Height(400), Width(352)]
        returns:
            4D tensor, float32, shape=(batch_size, new_chnnles, height, widht)
        """
        out = self.model(input)
        return out.reshape(*self.out_shape)


class RPN(nn.Module):
    def __init__(self, num_anchors_per_cell, requires_grad=True):
        super(RPN, self).__init__()
        self.num_anchors_per_cell = num_anchors_per_cell

        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU()
        ).to(device)
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(128).train(requires_grad), nn.ReLU()
        ).to(device)

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=(2, 2), padding = 1), nn.BatchNorm2d(256).train(requires_grad), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(256).train(requires_grad), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(256).train(requires_grad), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(256).train(requires_grad), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(256).train(requires_grad), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding = 1), nn.BatchNorm2d(256).train(requires_grad), nn.ReLU(),
        ).to(device)

        # Deconvolutions
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, (3, 3), stride=(1, 1), padding = 1), 
            nn.BatchNorm2d(256).train(requires_grad),
            nn.ReLU()
        ).to(device)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, (2, 2), stride=(2, 2), padding = 0), 
            nn.BatchNorm2d(256).train(requires_grad),
            nn.ReLU()
        ).to(device)
        self.deconv3 = nn.Sequential(
           nn.ConvTranspose2d(256, 256, (4, 4), stride=(4, 4), padding = 0), 
           nn.BatchNorm2d(256).train(requires_grad),
           nn.ReLU()
        ).to(device)

        # Probability and regression maps
        self.prob_map_conv = nn.Conv2d(768, self.num_anchors_per_cell, (1, 1), stride=(1, 1), padding = 0)
        self.reg_map_conv = nn.Conv2d(768, 7 * self.num_anchors_per_cell, (1, 1), stride=(1, 1), padding = 0)

        for p in self.parameters():
            p.requires_grad = requires_grad

        
    def forward(self, input):
        #input_shape = input.shape
        #assert len(input_shape) == 4 and input_shape[-1] % 8 == 0 and input_shape[-2] % 8 == 0, f"The input must be of shape [Batch_size, channels, map_height, map_width] with map_height and map_width multiple of 8, got {input_shape}"
        
        output      = self.block1(input)
        deconv1     = self.deconv1(output)
        
        output      = self.block2(output)
        deconv2     = self.deconv2(output)
        
        output      = self.block3(output)
        deconv3     = self.deconv3(output)
        output      = torch.cat([deconv3, deconv2, deconv1], dim=1)
        
        prob_map    = torch.sigmoid(self.prob_map_conv(output))
        reg_map     = self.reg_map_conv(output)
        
        return prob_map.permute(0, 2, 3, 1), reg_map.permute(0, 2, 3, 1) # TODO: Determine if this permute is correct.

class Model(nn.Module):
    def __init__(self, requires_grad=True):
        super(Model, self).__init__()
        self.nclasses = 1 #if type(cfg.DETECT_OBJECT) == str else len(cfg.DETECT_OBJECT)
        self.vfe_block = VFE_Block([32, 128], 128, [10, 200, 240], requires_grad=requires_grad).to(device)
        self.convMiddle = ConvMiddleLayer((1, 128, 200, 240), requires_grad=requires_grad).to(device)
        self.rpn = RPN(2, requires_grad=requires_grad).to(device)
        for p in self.parameters():
            p.requires_grad = requires_grad
            
    def forward(self, feature_buffer, coordinate_buffer):
        output = self.vfe_block(feature_buffer, coordinate_buffer)
        output = self.convMiddle(output)
        prob_map, reg_map = self.rpn(output)
        return prob_map, reg_map

