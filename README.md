This project presents an AI-powered College Bus Pass Management System that automates the process of issuing, verifying, and managing bus passes using facial recognition technology. It combines the power of Convolutional Neural Networks (CNN) with a user-friendly Flask-based web application to provide a secure, efficient, and contactless authentication system for students.

The system allows students to register on the website by submitting their facial images, which are then stored in a dataset and used to train a custom CNN model for facial recognition. At boarding points, the system captures real-time video frames using a webcam, detects faces using dlib and OpenCV, and verifies identity through the trained CNN model running on TensorFlow/Keras. A minimum confidence threshold of 70% is maintained to ensure reliability.

Key features of the system include digital ticket booking (one-way and return), automated expiration checks, digital receipts, secure login using facial recognition, and real-time feedback on user status. It also ensures data privacy through secure session tokens, encrypted credential storage, and API protection.

In performance tests, the system achieved:

Accuracy: 93.25%

Precision: 93.91%

Recall: 92.5%

F1-Score: 0.93

mAP (Mean Average Precision): 91.7%

These results demonstrate its high effectiveness in recognizing registered users while minimizing false positives and negatives.

The architecture was designed with scalability in mind, making it suitable for large campuses. The interface supports seamless integration with future features such as mobile apps, payment gateways, and live bus tracking.

This solution not only enhances operational efficiency but also introduces a contactless, AI-driven approach to managing student transportation, reducing manual intervention and ensuring a smooth experience for users.

# Facial-Recognition-Based-Bus-Pass-Verification-System-using-CNN
An AI-powered Bus Pass System using real-time facial recognition for student verification at boarding points. Built with Flask, TensorFlow, OpenCV, and dlib, it enables secure login, ticket booking, and real-time authentication with 93.25% accuracy.
