//
//  VGG16.swift
//  MPSCNN
//
//  Created by M.Ike on 2017/05/02.
//  Copyright © 2017年 M.Ike. All rights reserved.
//

import Foundation

import Foundation

class VGG16: CNNModel {
    // TODO: test
    var inputFormat = (w: 224, h: 224, ch: 3)
    func makeModel() -> [CNN.Layer] {
        return [
            conv2D("block1_conv1", 64, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("block1_conv2", 64, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("block1_pool", kernel: (2, 2), stride: (2, 2)),
            
            conv2D("block2_conv1", 128, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("block2_conv2", 128, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("block2_pool", kernel: (2, 2), stride: (2, 2)),
            
            conv2D("block3_conv1", 256, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("block3_conv2", 256, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("block3_conv3", 256, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("block3_pool", kernel: (2, 2), stride: (2, 2)),
            
            conv2D("block4_conv1", 512, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("block4_conv2", 512, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("block4_conv3", 512, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("block4_pool", kernel: (2, 2), stride: (2, 2)),
            
            conv2D("block5_conv1", 512, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("block5_conv2", 512, kernel: (3, 3), pad: .same, activation: .relu),
            conv2D("block5_conv3", 512, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("block5_pool", kernel: (2, 2), stride: (2, 2)),
            
            fully("fc1", 4096, activation: .relu),
            fully("fc2", 4096, activation: .relu),
            fully("predictions", 1000),
            softmax("softax")
        ]
    }
}
