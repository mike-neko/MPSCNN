//
//  ViewController.swift
//  MPSCNN
//
//  Created by M.Ike on 2017/04/29.
//  Copyright © 2017年 M.Ike. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        guard let cnn = CNN() else { return }
//        let mnist = MNIST()
//        cnn.setup(model: mnist)
//        let res = cnn.run(input: .image(convert2CGGray(image: UIImage(named: "3.png")!)!))
//        print(res)
        let vgg16 = VGG16()
        cnn.setup(model: vgg16)
        let res = cnn.run(input: .image(UIImage(named: "test.jpg")!)).sorted().reversed().first
        print(res)
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


    func convert2CGGray(image: UIImage) -> UIImage? {
        let rect = CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height)
        guard let context = CGContext(data: nil,
                                      width: Int(image.size.width),
                                      height: Int(image.size.height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue),
            let cgImage = image.cgImage  else {
                return nil
        }
        
        context.draw(cgImage, in: rect)
        return UIImage(cgImage: context.makeImage()!)
    }
}

