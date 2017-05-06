//
//  Inception.swift
//  MPSCNN
//
//  Created by M.Ike on 2017/05/05.
//  Copyright © 2017年 M.Ike. All rights reserved.
//

import Foundation

class Inception: CNNModel {
    // TODO: test
    var inputFormat = (w: 224, h: 224, ch: 3)
    func makeModel() -> [CNN.Layer] {
        return [
            conv2D("conv", 32, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu),
            conv2D("conv_1", 32, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("conv_2", 64, kernel: (3, 3), activation: .relu),
            conv2D("conv_3", 80, kernel: (1, 1), pad: .same, activation: .relu),
            conv2D("conv_4", 192, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("pool1", kernel: (3, 3), stride: (2, 2)),
            
            conv2D("mixed_conv", 64, kernel: (1, 1), activation: .relu),
            conv2D("mixed_tower_conv", 48, kernel: (1, 1), activation: .relu),
            conv2D("mixed_tower_conv_1", 64, kernel: (5, 5), activation: .relu),
            conv2D("mixed_tower_1_conv", 64, kernel: (1, 1), activation: .relu),
            conv2D("mixed_tower_1_conv_1", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_tower_1_conv_2", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_tower_2_conv", 32, kernel: (1, 1), activation: .relu),

            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
            
            conv2D("mixed_1_conv", 64, kernel: (1, 1), activation: .relu),
            conv2D("mixed_1_tower_conv", 48, kernel: (1, 1), activation: .relu),
            conv2D("mixed_1_tower_conv_1", 64, kernel: (5, 5), activation: .relu),
            conv2D("mixed_1_tower_1_conv", 64, kernel: (1, 1), activation: .relu),
            conv2D("mixed_1_tower_1_conv_1", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_1_tower_1_conv_2", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_1_tower_2_conv", 64, kernel: (1, 1), activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
            
            conv2D("mixed_2_conv", 64, kernel: (1, 1), activation: .relu),
            conv2D("mixed_2_tower_conv", 48, kernel: (1, 1), activation: .relu),
            conv2D("mixed_2_tower_conv_1", 64, kernel: (5, 5), activation: .relu),
            conv2D("mixed_2_tower_1_conv", 64, kernel: (1, 1), activation: .relu),
            conv2D("mixed_2_tower_1_conv_1", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_2_tower_1_conv_2", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_2_tower_2_conv", 64, kernel: (1, 1), activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
            
            conv2D("mixed_3_conv", 384, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu),
            conv2D("mixed_3_tower_conv", 64, kernel: (1, 1), activation: .relu),
            conv2D("mixed_3_tower_conv_1", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_3_tower_conv_2", 96, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),       // FIXME: dest
            
            conv2D("mixed_4_conv", 192, kernel: (1, 1), activation: .relu),
            conv2D("mixed_4_tower_conv", 128, kernel: (1, 1), activation: .relu),
            conv2D("mixed_4_tower_conv_1", 128, kernel: (7, 1), activation: .relu),
            conv2D("mixed_4_tower_conv_2", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_4_tower_1_conv", 128, kernel: (1, 1), activation: .relu),
            conv2D("mixed_4_tower_1_conv_1", 128, kernel: (1, 7), activation: .relu),
            conv2D("mixed_4_tower_1_conv_2", 128, kernel: (7, 1), activation: .relu),
            conv2D("mixed_4_tower_1_conv_3", 128, kernel: (1, 7), activation: .relu),
            conv2D("mixed_4_tower_1_conv_4", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_4_tower_2_conv", 192, kernel: (1, 1), activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
            
            conv2D("mixed_5_conv", 192, kernel: (1, 1), activation: .relu),
            conv2D("mixed_5_tower_conv", 160, kernel: (1, 1), activation: .relu),
            conv2D("mixed_5_tower_conv_1", 160, kernel: (7, 1), activation: .relu),
            conv2D("mixed_5_tower_conv_2", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_5_tower_1_conv", 160, kernel: (1, 1), activation: .relu),
            conv2D("mixed_5_tower_1_conv_1", 160, kernel: (1, 7), activation: .relu),
            conv2D("mixed_5_tower_1_conv_2", 160, kernel: (7, 1), activation: .relu),
            conv2D("mixed_5_tower_1_conv_3", 160, kernel: (1, 7), activation: .relu),
            conv2D("mixed_5_tower_1_conv_4", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_5_tower_2_conv", 192, kernel: (1, 1), activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
            
            conv2D("mixed_6_conv", 192, kernel: (1, 1), activation: .relu),
            conv2D("mixed_6_tower_conv", 160, kernel: (1, 1), activation: .relu),
            conv2D("mixed_6_tower_conv_1", 160, kernel: (7, 1), activation: .relu),
            conv2D("mixed_6_tower_conv_2", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_6_tower_1_conv", 160, kernel: (1, 1), activation: .relu),
            conv2D("mixed_6_tower_1_conv_1", 160, kernel: (1, 7), activation: .relu),
            conv2D("mixed_6_tower_1_conv_2", 160, kernel: (7, 1), activation: .relu),
            conv2D("mixed_6_tower_1_conv_3", 160, kernel: (1, 7), activation: .relu),
            conv2D("mixed_6_tower_1_conv_4", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_6_tower_2_conv", 192, kernel: (1, 1), activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
           
            conv2D("mixed_7_conv", 192, kernel: (1, 1), activation: .relu),
            conv2D("mixed_7_tower_conv", 192, kernel: (1, 1), activation: .relu),
            conv2D("mixed_7_tower_conv_1", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_7_tower_conv_2", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_7_tower_1_conv", 192, kernel: (1, 1), activation: .relu),
            conv2D("mixed_7_tower_1_conv_1", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_7_tower_1_conv_2", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_7_tower_1_conv_3", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_7_tower_1_conv_4", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_7_tower_2_conv", 192, kernel: (1, 1), activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
            
            conv2D("mixed_8_tower_conv", 192, kernel: (1, 1), activation: .relu),
            conv2D("mixed_8_tower_conv_1", 320, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu),
            conv2D("mixed_8_tower_1_conv", 192, kernel: (1, 1), activation: .relu),
            conv2D("mixed_8_tower_1_conv_1", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_8_tower_1_conv_2", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_8_tower_1_conv_3", 192, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (2, 2)),       // FIXME: dest
            
            conv2D("mixed_9_conv", 320, kernel: (1, 1), activation: .relu),
            conv2D("mixed_9_tower_conv", 384, kernel: (1, 1), activation: .relu),
            conv2D("mixed_9_tower_mixed_conv", 384, kernel: (3, 1), activation: .relu),
            conv2D("mixed_9_tower_mixed_conv_1", 384, kernel: (1, 3), activation: .relu),
            conv2D("mixed_9_tower_1_conv", 448, kernel: (1, 1), activation: .relu),
            conv2D("mixed_9_tower_1_conv_1", 384, kernel: (3, 3), activation: .relu),
            conv2D("mixed_9_tower_1_mixed_conv", 384, kernel: (3, 1), activation: .relu),
            conv2D("mixed_9_tower_1_mixed_conv_1", 384, kernel: (1, 3), activation: .relu),
            conv2D("mixed_9_tower_2_conv", 192, kernel: (1, 1), activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
            
            conv2D("mixed_10_conv", 320, kernel: (1, 1), activation: .relu),
            conv2D("mixed_10_tower_conv", 384, kernel: (1, 1), activation: .relu),
            conv2D("mixed_10_tower_mixed_conv", 384, kernel: (3, 1), activation: .relu),
            conv2D("mixed_10_tower_mixed_conv_1", 384, kernel: (1, 3), activation: .relu),
            conv2D("mixed_10_tower_1_conv", 448, kernel: (1, 1), activation: .relu),
            conv2D("mixed_10_tower_1_conv_1", 384, kernel: (3, 3), activation: .relu),
            conv2D("mixed_10_tower_1_mixed_conv", 384, kernel: (3, 1), activation: .relu),
            conv2D("mixed_10_tower_1_mixed_conv_1", 384, kernel: (1, 3), activation: .relu),
            conv2D("mixed_10_tower_2_conv", 192, kernel: (1, 1), activation: .relu),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),
            
            maxPooling2D("", kernel: (3, 3), stride: (1, 1)),   // FIXME:
            

            fully("predictions", 1008),
            softmax("softax")
        ]
    }
}
