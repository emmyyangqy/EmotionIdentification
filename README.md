# EmotionIdentification

This code utilizes MediaPipe Face Mesh, a real-time 3D face landmark estimation technology, to extract the 3D coordinates of the 468 face landmarks estimated by the Face Mesh and uses it in the calculation to generate a heatmap to represent the facial expression. These heatmaps are saved in labeled emotion folders (depending on the facial expressions of the origal image) and are used to train a CNN model.

("data" folder must be created)
