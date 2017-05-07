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
            maxPooling2D("pool", kernel: (3, 3), stride: (2, 2)),
            conv2D("conv_3", 80, kernel: (1, 1), pad: .same, activation: .relu),
            conv2D("conv_4", 192, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("pool_1", kernel: (3, 3), stride: (2, 2)),
            
            conv2D("mixed_conv", 64, kernel: (1, 1), activation: .relu, input: "pool_1", output: "image0"),
            conv2D("mixed_tower_conv", 48, kernel: (1, 1), activation: .relu, input: "pool_1"),
            conv2D("mixed_tower_conv_1", 64, kernel: (5, 5), activation: .relu, output: "image0"),
            conv2D("mixed_tower_1_conv", 64, kernel: (1, 1), activation: .relu, input: "pool_1"),
            conv2D("mixed_tower_1_conv_1", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_tower_1_conv_2", 96, kernel: (3, 3), activation: .relu, output: "image0"),
            maxPooling2D("pool_2", kernel: (3, 3), stride: (1, 1), input: "pool_1"),
            conv2D("mixed_tower_2_conv", 32, kernel: (1, 1), activation: .relu, output: "image0"),
    
            conv2D("mixed_1_conv", 64, kernel: (1, 1), activation: .relu, input: "image0", output: "image1"),
            conv2D("mixed_1_tower_conv", 48, kernel: (1, 1), activation: .relu, input: "image0"),
            conv2D("mixed_1_tower_conv_1", 64, kernel: (5, 5), activation: .relu, output: "image1"),
            conv2D("mixed_1_tower_1_conv", 64, kernel: (1, 1), activation: .relu, input: "image0"),
            conv2D("mixed_1_tower_1_conv_1", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_1_tower_1_conv_2", 96, kernel: (3, 3), activation: .relu, output: "image1"),
            maxPooling2D("pool1", kernel: (3, 3), stride: (1, 1), input: "image0"),
            conv2D("mixed_1_tower_2_conv", 64, kernel: (1, 1), activation: .relu, output: "image1"),
            
            conv2D("mixed_2_conv", 64, kernel: (1, 1), activation: .relu, input: "image1", output: "image2"),
            conv2D("mixed_2_tower_conv", 48, kernel: (1, 1), activation: .relu, input: "image1"),
            conv2D("mixed_2_tower_conv_1", 64, kernel: (5, 5), activation: .relu, output: "image2"),
            conv2D("mixed_2_tower_1_conv", 64, kernel: (1, 1), activation: .relu, input: "image1"),
            conv2D("mixed_2_tower_1_conv_1", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_2_tower_1_conv_2", 96, kernel: (3, 3), activation: .relu, output: "image2"),
            maxPooling2D("pool2", kernel: (3, 3), stride: (1, 1), input: "image1"),
            conv2D("mixed_2_tower_2_conv", 64, kernel: (1, 1), activation: .relu, output: "image2"),
            
            conv2D("mixed_3_conv", 384, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu, input: "image2", output: "image3"),
            conv2D("mixed_3_tower_conv", 64, kernel: (1, 1), activation: .relu, input: "image2"),
            conv2D("mixed_3_tower_conv_1", 96, kernel: (3, 3), activation: .relu),
            conv2D("mixed_3_tower_conv_2", 96, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu, output: "image3"),
            maxPooling2D("pool3", kernel: (3, 3), stride: (2, 2), input: "image2", output: "image3"),       // FIXME: dest
            
            conv2D("mixed_4_conv", 192, kernel: (1, 1), activation: .relu, input: "image3", output: "image4"),
            conv2D("mixed_4_tower_conv", 128, kernel: (1, 1), activation: .relu, input: "image3"),
            conv2D("mixed_4_tower_conv_1", 128, kernel: (7, 1), activation: .relu),
            conv2D("mixed_4_tower_conv_2", 192, kernel: (1, 7), activation: .relu, output: "image4"),
            conv2D("mixed_4_tower_1_conv", 128, kernel: (1, 1), activation: .relu, input: "image3"),
            conv2D("mixed_4_tower_1_conv_1", 128, kernel: (1, 7), activation: .relu),
            conv2D("mixed_4_tower_1_conv_2", 128, kernel: (7, 1), activation: .relu),
            conv2D("mixed_4_tower_1_conv_3", 128, kernel: (1, 7), activation: .relu),
            conv2D("mixed_4_tower_1_conv_4", 192, kernel: (7, 1), activation: .relu, output: "image4"),
            maxPooling2D("pool4", kernel: (3, 3), stride: (1, 1), input: "image3"),
            conv2D("mixed_4_tower_2_conv", 192, kernel: (1, 1), activation: .relu, output: "image4"),

            conv2D("mixed_5_conv", 192, kernel: (1, 1), activation: .relu, input: "image4", output: "image5"),
            conv2D("mixed_5_tower_conv", 160, kernel: (1, 1), activation: .relu, input: "image4"),
            conv2D("mixed_5_tower_conv_1", 160, kernel: (7, 1), activation: .relu),
            conv2D("mixed_5_tower_conv_2", 192, kernel: (1, 7), activation: .relu, output: "image5"),
            conv2D("mixed_5_tower_1_conv", 160, kernel: (1, 1), activation: .relu, input: "image4"),
            conv2D("mixed_5_tower_1_conv_1", 160, kernel: (1, 7), activation: .relu),
            conv2D("mixed_5_tower_1_conv_2", 160, kernel: (7, 1), activation: .relu),
            conv2D("mixed_5_tower_1_conv_3", 160, kernel: (1, 7), activation: .relu),
            conv2D("mixed_5_tower_1_conv_4", 192, kernel: (7, 1), activation: .relu, output: "image5"),
            maxPooling2D("pool5", kernel: (3, 3), stride: (1, 1), input: "image4"),
            conv2D("mixed_5_tower_2_conv", 192, kernel: (1, 1), activation: .relu, output: "image5"),
            
            conv2D("mixed_6_conv", 192, kernel: (1, 1), activation: .relu, input: "image5", output: "image6"),
            conv2D("mixed_6_tower_conv", 160, kernel: (1, 1), activation: .relu, input: "image5"),
            conv2D("mixed_6_tower_conv_1", 160, kernel: (7, 1), activation: .relu),
            conv2D("mixed_6_tower_conv_2", 192, kernel: (1, 7), activation: .relu, output: "image6"),
            conv2D("mixed_6_tower_1_conv", 160, kernel: (1, 1), activation: .relu, input: "image5"),
            conv2D("mixed_6_tower_1_conv_1", 160, kernel: (1, 7), activation: .relu),
            conv2D("mixed_6_tower_1_conv_2", 160, kernel: (7, 1), activation: .relu),
            conv2D("mixed_6_tower_1_conv_3", 160, kernel: (1, 7), activation: .relu),
            conv2D("mixed_6_tower_1_conv_4", 192, kernel: (7, 1), activation: .relu, output: "image6"),
            maxPooling2D("pool6", kernel: (3, 3), stride: (1, 1), input: "image5"),
            conv2D("mixed_6_tower_2_conv", 192, kernel: (1, 1), activation: .relu, output: "image6"),
            
            conv2D("mixed_7_conv", 192, kernel: (1, 1), activation: .relu, input: "image6", output: "image7"),
            conv2D("mixed_7_tower_conv", 192, kernel: (1, 1), activation: .relu, input: "image6"),
            conv2D("mixed_7_tower_conv_1", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_7_tower_conv_2", 192, kernel: (1, 7), activation: .relu, output: "image7"),
            conv2D("mixed_7_tower_1_conv", 192, kernel: (1, 1), activation: .relu, input: "image6"),
            conv2D("mixed_7_tower_1_conv_1", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_7_tower_1_conv_2", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_7_tower_1_conv_3", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_7_tower_1_conv_4", 192, kernel: (7, 1), activation: .relu, output: "image7"),
            maxPooling2D("pool7", kernel: (3, 3), stride: (1, 1), input: "image6"),
            conv2D("mixed_7_tower_2_conv", 192, kernel: (1, 1), activation: .relu, output: "image7"),

            conv2D("mixed_8_tower_conv", 192, kernel: (1, 1), activation: .relu, input: "image7", output: "image8"),
            conv2D("mixed_8_tower_conv_1", 320, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu),
            conv2D("mixed_8_tower_1_conv", 192, kernel: (1, 1), activation: .relu, input: "image7"),
            conv2D("mixed_8_tower_1_conv_1", 192, kernel: (7, 1), activation: .relu),
            conv2D("mixed_8_tower_1_conv_2", 192, kernel: (1, 7), activation: .relu),
            conv2D("mixed_8_tower_1_conv_3", 192, kernel: (3, 3), stride: (2, 2), pad: .same, activation: .relu, output: "image8"),
            maxPooling2D("pool8", kernel: (3, 3), stride: (2, 2), input: "image7", output: "image8"),       // FIXME: dest
            
            conv2D("mixed_9_conv", 320, kernel: (1, 1), activation: .relu, input: "image8", output: "image9"),
            conv2D("mixed_9_tower_conv", 384, kernel: (1, 1), activation: .relu, input: "image8", output: "image9_1"),
            conv2D("mixed_9_tower_mixed_conv", 384, kernel: (3, 1), activation: .relu, input: "image9_1", output: "image9"),
            conv2D("mixed_9_tower_mixed_conv_1", 384, kernel: (1, 3), activation: .relu, input: "image9_1", output: "image9"),
            conv2D("mixed_9_tower_1_conv", 448, kernel: (1, 1), activation: .relu, input: "image8"),
            conv2D("mixed_9_tower_1_conv_1", 384, kernel: (3, 3), activation: .relu, output: "image9_2"),
            conv2D("mixed_9_tower_1_mixed_conv", 384, kernel: (3, 1), activation: .relu, input: "image9_2", output: "image9"),
            conv2D("mixed_9_tower_1_mixed_conv_1", 384, kernel: (1, 3), activation: .relu, input: "image9_2", output: "image9"),
            maxPooling2D("pool9", kernel: (3, 3), stride: (1, 1), input: "image8"),
            conv2D("mixed_9_tower_2_conv", 192, kernel: (1, 1), activation: .relu, output: "image9"),

            conv2D("mixed_10_conv", 320, kernel: (1, 1), activation: .relu, input: "image9", output: "image10"),
            conv2D("mixed_10_tower_conv", 384, kernel: (1, 1), activation: .relu, input: "image9", output: "image10_1"),
            conv2D("mixed_10_tower_mixed_conv", 384, kernel: (3, 1), activation: .relu, input: "image10_1", output: "image10"),
            conv2D("mixed_10_tower_mixed_conv_1", 384, kernel: (1, 3), activation: .relu, input: "image10_1", output: "image10"),
            conv2D("mixed_10_tower_1_conv", 448, kernel: (1, 1), activation: .relu, input: "image9"),
            conv2D("mixed_10_tower_1_conv_1", 384, kernel: (3, 3), activation: .relu, output: "image10_2"),
            conv2D("mixed_10_tower_1_mixed_conv", 384, kernel: (3, 1), activation: .relu, input: "image10_2", output: "image10"),
            conv2D("mixed_10_tower_1_mixed_conv_1", 384, kernel: (1, 3), activation: .relu, input: "image10_2", output: "image10"),
            maxPooling2D("pool10", kernel: (3, 3), stride: (1, 1), input: "image9"),
            conv2D("mixed_10_tower_2_conv", 192, kernel: (1, 1), activation: .relu, output: "image10"),

            averagePooling2D("pool_logits", kernel: (8, 8), stride: (4, 4), input: "image10"),

            fully("predictions", 1008),
            softmax("softax")
        ]
    }
}
