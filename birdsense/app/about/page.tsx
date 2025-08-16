"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Bird,
  Camera,
  Cpu,
  Code,
  Database,
  CheckCircle,
  ExternalLink,
  Cloud,
  Server,
} from "lucide-react"
import AppLayout from "@/components/app-layout"

export default function HowItWorksPage() {
  const tableOfContents = [
    { id: "introduction", title: "Project Introduction", icon: Bird },
    { id: "architecture", title: "System Architecture", icon: Cpu },
    { id: "hardware", title: "Hardware Setup", icon: Camera },
    { id: "machine-learning", title: "Machine Learning Model", icon: Code },
    { id: "cloud-infrastructure", title: "Cloud Infrastructure", icon: Cloud },
  ]

  const technologies = [
    { name: "Raspberry Pi 5", category: "Hardware", color: "bg-red-100 text-red-800" },
    { name: "Python", category: "Backend", color: "bg-blue-100 text-blue-800" },
    { name: "Swin Transformer", category: "ML Model", color: "bg-orange-100 text-orange-800" },
    { name: "YOLO11", category: "ML Model", color: "bg-orange-100 text-orange-800" },
    { name: "AWS Lambda", category: "Serverless", color: "bg-yellow-100 text-yellow-800" },
    { name: "S3 bucket", category: "Serverless", color: "bg-yellow-100 text-yellow-800" },
    { name: "MongoDB Atlas", category: "Database", color: "bg-green-100 text-green-800" },
    { name: "Next.js", category: "Frontend", color: "bg-gray-100 text-gray-800" },
    { name: "TypeScript", category: "Language", color: "bg-blue-100 text-blue-800" },
    { name: "OpenCV", category: "Computer Vision", color: "bg-green-100 text-green-800" },
    { name: "PyTorch", category: "ML Framework", color: "bg-red-100 text-red-800" },
  ]

  const scrollToSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <AppLayout showNavigation={true}>
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hello Section */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="relative">
              <div className="w-24 h-24 bg-gradient-to-br from-emerald-500 to-blue-600 rounded-full flex items-center justify-center shadow-xl">
                <Bird className="w-12 h-12 text-white" />
              </div>
              <div className="absolute -bottom-2 -right-2 w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center">
                <Code className="w-4 h-4 text-white" />
              </div>
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            How <span className="text-emerald-600">birdSense</span> Works
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
            Read about how I built a AI powered bird classification system for my backyard. Built using a Raspberry 
            Pi, Swin transformer model, aws cloud infrastructure, and this Next.js web application.
          </p>
        </div>

        {/* Table of Contents */}
        <Card className="mb-12 bg-gradient-to-r from-emerald-50 to-blue-50 border-emerald-200">
          <CardContent className="p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
              <Database className="w-5 h-5 mr-2 text-emerald-600" />
              Table of Contents
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {tableOfContents.map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className="flex items-center space-x-3 p-3 rounded-lg hover:bg-white/60 transition-colors text-left"
                >
                  <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center">
                    <item.icon className="w-4 h-4 text-emerald-600" />
                  </div>
                  <span className="font-medium text-gray-700">{item.title}</span>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Introduction Section */}
        <section id="introduction" className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center">
            <Bird className="w-8 h-8 mr-3 text-emerald-600" />
            Project Introduction
          </h2>

          <div className="prose prose-lg max-w-none mb-8">
            <p className="text-gray-700 leading-relaxed mb-6">
              I decided I wanted to build BirdSense to help my dad identify all the different specicies of 
              birds that visit our bird feeders in our backyard. I also wanted to build something with machine learning and computer vision
              and those things interst me and I wanted to learn more about them. what begain as simple curioisity quickly evolved into this fully fleged project, invloving reading research papers, 
              training my own neural network, using a raspberry pi and through that learning about linux, and desinigng a website. I had no idea how challanging but also
              how rewarding it would be. I learned a emense amount about machine learning, computer vision, system architecture, and cloud infasturture. The end result is more than anything I could have 
              imagined I could have built. The model achieves roughly 92% accuracy on over 200 species of birds. 

            </p>

            <div className="bg-emerald-50 border-l-4 border-emerald-500 p-6 mb-6">
              <h3 className="text-lg font-semibold text-emerald-800 mb-2">Project Goals</h3>
              <ul className="space-y-2 text-emerald-700">
                <li className="flex items-center">
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Real-time bird species identification using ML
                </li>
                <li className="flex items-center">
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Scalable cloud architecture
                </li>
                <li className="flex items-center">
                  <CheckCircle className="w-4 h-4 mr-2" />
                  user friendly web application to display bird classifications
                </li>
              </ul>
            </div>
          </div>

          <Card className="mb-8">
            <CardContent className="p-6">
              <img
                src="home-page.png?height=400&width=800&text=birdSense+Cloud+Architecture+Overview"
                alt="BirdSense Home Page"
                className="w-full rounded-lg mb-4"
              />
              <p className="text-sm text-gray-600 text-center italic">
                BirdSense Home page 
              </p>
            </CardContent>
          </Card>
        </section>

        {/* Architecture Section */}
        <section id="architecture" className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center">
            <Cpu className="w-8 h-8 mr-3 text-blue-600" />
            System Architecture
          </h2>

          <div className="prose prose-lg max-w-none mb-8">
            <p className="text-gray-700 leading-relaxed mb-6">
              The birdSense architecture implements a hybrid edge-cloud model, optimizing for both real-time
              performance and scalable processing. The system uses a raspberry pi, AWS Lambda for serverless functions, MongoDB Atlas
              for data storage, and Next.js for a responsive web experience.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <Card className="border-blue-200">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <Camera className="w-6 h-6 text-blue-600 mr-2" />
                  <h3 className="text-lg font-semibold">Edge Processing Layer</h3>
                </div>
                <ul className="space-y-2 text-gray-600">
                  <li>• Raspberry Pi with camera module</li>
                  <li>• YOlO11 for detection</li>
                  <li>• SwinT model for classification</li>
                  <li>• norfair object tracking</li>
                  <li>• upload via API</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-yellow-200">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <Cloud className="w-6 h-6 text-yellow-600 mr-2" />
                  <h3 className="text-lg font-semibold">Cloud Processing Layer</h3>
                </div>
                <ul className="space-y-2 text-gray-600">
                  <li>• AWS Lambda serverless functions</li>
                  <li>• MongoDB Atlas </li>
                  <li>• Authentication</li>
                </ul>
              </CardContent>
            </Card>
          </div>

          <Card className="mb-8">
            <CardContent className="p-6">
              <img
                src="/system-overview.png?height=300&width=800&text=Hybrid+Edge-Cloud+Architecture"
                alt="System Architecture Diagram"
                className="w-full h-48 object-cover rounded-lg mb-4"
              />
              <p className="text-sm text-gray-600 text-center italic">
                Data flow from edge device through AWS Lambda to MongoDB Atlas and Next.js frontend
              </p>
            </CardContent>
          </Card>

          {/* Technologies Used */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Technologies Used</h3>
            <div className="flex flex-wrap gap-2">
              {technologies.map((tech, index) => (
                <Badge key={index} className={`${tech.color} border-0 px-3 py-1`}>
                  {tech.name}
                </Badge>
              ))}
            </div>
          </div>
        </section>

        {/* Hardware Setup Section */}
        <section id="hardware" className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center">
            <Camera className="w-8 h-8 mr-3 text-purple-600" />
            Hardware Setup
          </h2>

          <div className="prose prose-lg max-w-none mb-8">
            <p className="text-gray-700 leading-relaxed mb-6">
              The edge computing component uses a Raspberry Pi 5 as the primary processing unit. A rasbperry pi camera modeule 3 
              is used for the camrea. Designed 3D printed housing for raspberry pi and mount for camera (waiting on 3d-printer access to print). 
              The hardware setup focuses on reliability, performance, and weather resistance. 
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Hardware Components</h3>
              <div className="space-y-4">
                <Card className="border-l-4 border-l-purple-500">
                  <CardContent className="p-4">
                    <h4 className="font-semibold text-gray-900">Raspberry Pi 5 (16GB RAM)</h4>
                    <p className="text-sm text-gray-600">memory for ML operations</p>
                  </CardContent>
                </Card>
                <Card className="border-l-4 border-l-blue-500">
                  <CardContent className="p-4">
                    <h4 className="font-semibold text-gray-900">Raspberry pi Camera module 3</h4>
                    <p className="text-sm text-gray-600">12MP sensor with autofocus</p>
                  </CardContent>
                </Card>
                <Card className="border-l-4 border-l-green-500">
                  <CardContent className="p-4">
                    <h4 className="font-semibold text-gray-900">Enclosure</h4>
                    <p className="text-sm text-gray-600">housing and camera mount with thermal management</p>
                  </CardContent>
                </Card>
                <Card className="border-l-4 border-l-orange-500">
                  <CardContent className="p-4">
                    <h4 className="font-semibold text-gray-900">2.4 Ghz WiFi Connectivity</h4>
                    <p className="text-sm text-gray-600">connected to internet even in backyard to allow uploading to backend</p>
                  </CardContent>
                </Card>
              </div>
            </div>

            <Card>
              <CardContent className="p-6">
                <img
                  src="/placeholder.png?height=300&width=400&text=Edge+Device+Setup"
                  alt="Edge Device Hardware Setup"
                  className="w-full h-48 object-cover rounded-lg mb-4"
                />
                <p className="text-sm text-gray-600 text-center italic">
                  Rapberry Pi with camera In housing
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Machine Learning Section */}
        <section id="machine-learning" className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center">
            <Code className="w-8 h-8 mr-3 text-orange-600" />
            Machine Learning Pipeline
          </h2>

          <div className="prose prose-lg max-w-none mb-8">
            <p className="text-gray-700 leading-relaxed mb-6">
              To balance performance and accuract, BirdSense deploys a two-stage pipeline. First YOLOV11 detects and birds in the camera frame and draw a bounding box around them, then the cropped image is fed into the Swin transformer 
              model for classification. 
              <br></br><br></br>

              At first I wanted to use a dataset provided by Cornell lab of ornithology to train my classification model.
              I tried to use their NABirds dataset, this contained classification images for 400 species of birds, and was hierarchical. I quickly realized that 
              that I was way above my head and that I needed a simplier dataset, without a hierarchical structure. I then discovered the dataset CUB-200, a much simplier dataset with 200 species of birds and over 10000 images 
              which I ended up using for training. I split the dataset into 50% training and 50% validation according to metadata provided by the dataset. 

              <br></br><br></br>
              To figure out how to train my model I read several research papers that utilized the CUB-200 dataset. I found the paper: <a 
                href="https://arxiv.org/abs/2303.06442" 
                className="text-blue-600 hover:text-blue-800 hover:underline"
                target="_blank" 
                rel="noopener noreferrer" >Fine-grained Visual Classification with High-temperature Refinement and Background Suppression</a> interesting and decided to use it as a model for my training pipeline.
              I initialized the SwinT backbone using the ImageNet pretrained weights. The image size was 384x384, I used an effective batch size of 32, 80 training epochs and a LR scheduler with a linear warmup followed 
              by cosine annealing to a LR of zero. I used augmentation of random resize crop, random erasing, and the built in random augmentation method with pyTorch. the validation accuracy was logged into TensorBoard at each epoch
              and the best model checkpoint was saved whenever accuracy improved. The classification head combined the pre-trained backbone with a BSHead that computes top k-pooling and composite loss to achieve background suppression as specified in the paper. 
              The model was trained using PyTorch on my Desktop computer with a 3070 Super, it took 3 whole days to run all 80 epochs. In the end the classifier achieved a top accuracy of 92% on the validation set.  



            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Pipeline</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                      <span className="text-orange-600 font-bold text-sm">1</span>
                    </div>
                    <span className="text-gray-700">Detect birds in frame (YOLOv11)</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                      <span className="text-orange-600 font-bold text-sm">2</span>
                    </div>
                    <span className="text-gray-700">crop by bounding box to isolate bird</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                      <span className="text-orange-600 font-bold text-sm">3</span>
                    </div>
                    <span className="text-gray-700">Classify isolated bird image with SwinT + BShead</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                      <span className="text-orange-600 font-bold text-sm">4</span>
                    </div>
                    <span className="text-gray-700">Upload classification via lambda and display </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-1">
                <img
                  src="/graphs.png?height=250&width=400&text=Swin+Transformer+Architecture"
                  alt="SwinT training"
                  className="w-full h-full"
                />
                <p className="text-sm text-gray-600 text-center italic">
                  SwinT training
                </p>
              </CardContent>
            </Card>
          </div>

          <Card className="mb-8 bg-gray-50">
  <CardContent className="p-6">
    <h3 className="text-lg font-semibold text-gray-900 mb-6 text-center">Model Performance Metrics</h3>
    <div className="grid grid-cols-2 gap-8 max-w-lg mx-auto">
      <div className="text-center">
        <div className="text-4xl font-bold text-emerald-600 mb-2">92%</div>
        <div className="text-sm text-gray-600">Top-1 Accuracy</div>
      </div>
      <div className="text-center">
        <div className="text-4xl font-bold text-blue-600 mb-2">200</div>
        <div className="text-sm text-gray-600">Species Classes</div>
      </div>
    </div>
  </CardContent>
</Card>
        </section>

        {/* Cloud Infrastructure Section */}
        <section id="cloud-infrastructure" className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center">
            <Cloud className="w-8 h-8 mr-3 text-yellow-600" />
            Cloud Infrastructure
          </h2>

          <div className="prose prose-lg max-w-none mb-8">
            <p className="text-gray-700 leading-relaxed mb-6">
              I decided to use AWS and MongoDB Atlas for cloud infrastructure because of their scability, free to use tier, and because I wanter to learn how to use them. 
              the cloud infrastructure handles my backend API and datastorage needs. I have a lambda functions that generates presigned URLs for loading and uploading images from my s3 bucket, and I have 
              lambda functions that handle uploading and loading classification data to a from my MonogDB database. I also have a lambda function that handles my authentication. This setup is easy to maintain, extremely scalable, free, and extremely reliable. 
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <Card className="border-yellow-200">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <Server className="w-6 h-6 text-yellow-600 mr-2" />
                  <h3 className="text-lg font-semibold">AWS Lambda Functions</h3>
                </div>
                <ul className="space-y-2 text-gray-600">
                  <li>• Generate S3 pre-signed URLs for the Pi to upload cropped bird images</li>
                  <li>• Receive JSON payloads from the Pi containing species, confidence, timestamp, and S3 key</li>
                  <li>• Write images into S3 via the signed URL; hand off metadata to MongoDB Atlas</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-green-200">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <Database className="w-6 h-6 text-green-600 mr-2" />
                  <h3 className="text-lg font-semibold">MongoDB Atlas</h3>
                </div>
                <ul className="space-y-2 text-gray-600">
                  <li>• Persist each classification record (species, confidence score, timestamp, S3 image key)</li>
                  <li>• User session and device management</li>
                  <li>• Support real-time queries for the Next.js front-end</li>
                  <li>• Index on timestamp and species to power analytics and history lookups</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>
                {/*GitHub Link */}
        <div className="text-center mb-8">
          <a
            href="https://github.com/FelixjGrimm118/BirdSense"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-2 text-emerald-600 hover:text-emerald-700 transition-colors duration-200 text-lg font-medium group"
          >
            <svg
              className="w-5 h-5 group-hover:scale-110 transition-transform duration-200"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
            </svg>
            <span className="border-b border-emerald-300 group-hover:border-emerald-500 transition-colors duration-200">
              View Github
            </span>
            <ExternalLink className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-200" />
          </a>
        </div>
      </div>
    </AppLayout>
  )
}
