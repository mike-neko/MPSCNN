//
//  MNIST.swift
//  MPSCNN
//
//  Created by M.Ike on 2017/04/30.
//  Copyright © 2017年 M.Ike. All rights reserved.
//

import Foundation

class MNIST: CNNModel {
    // TODO: test
    var inputFormat = (w: 28, h: 28, ch: 1)
    func makeModel() -> [CNN.Layer] {
        return [
            conv2D("conv1", 32, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("pool1", kernel: (2, 2), stride: (2, 2)),
            conv2D("conv2", 64, kernel: (3, 3), pad: .same, activation: .relu),
            maxPooling2D("pool2", kernel: (2, 2), stride: (2, 2)),
            fully("fc1", 1024),
            fully("fc2", 10),
            softmax("softax")
        ]
    }
}
