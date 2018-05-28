package main;

import org.opencv.core.Mat;

import java.util.LinkedList;

public class FaceRecognition {

    private static boolean trained;
    private static LinkedList<Mat> faces = new LinkedList<>();
    private static double[] theta = new double[1600];

    public static boolean trainFace(Mat grayface) {
        if (!trained){
            if(faces.size() == 40){
                trained = true;
                Classification.addFaces(faces);
                System.out.println("[ 100% ]Face Training Completed");
                theta = Classification.learn(theta);
                return true;
            }
            faces.add(grayface);
        }
        return false;
    }

    public static double checkFace(Mat grayface) {
        int colCounter = 0;
        double g = 0;
        for (int i = 0; i < grayface.rows(); i++) {
            for (int j = 0; j < grayface.cols(); j++) {
                 g += grayface.get(i, j)[0] * theta[colCounter++];
            }
        }
        return Classification.sigmoid(g);
    }
}
