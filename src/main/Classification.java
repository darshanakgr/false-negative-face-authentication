package main;

import org.opencv.core.Mat;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.LinkedList;

public class Classification {

    private static double[][] X = new double[160][1600];
    private static double[] Y = new double[160];
    private static short counter = 120;

    static {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./src/img/database.dat"))) {
            double[][] faces = (double[][]) ois.readObject();
            addFaces(faces);
        } catch (IOException e) {
            System.err.println("Cannot find the database file: " + e);
        } catch (ClassNotFoundException e) {
            System.err.println("Casting error: " + e);
        } finally {
            for (int i = 120; i < 160; i++) {
                Y[i] = 1;
            }
        }
    }

    public static double[] learn(double[] theta) {
        return learn(theta, X, Y);
    }

    public static double[] learn(double[] theta, double[][] x, double[] y) {
        for (int i = 0; i < 500; i++) {
            System.out.printf("[ GRDES ] Iteration %2d: %.9f\n", i, costFunction(theta, x, y));
            double[] grad = grad(theta, x, y);
            for (int j = 0; j < theta.length; j++) {
                theta[j] = theta[j] - (0.001 / y.length) * grad[j];
            }
        }
        return theta;
    }

    private static double costFunction(double[] theta, double[][] x, double[] y) {
        double J = 0;
        int m = y.length;
        for (int i = 0; i < y.length; i++) {
            double sum = 0;
            for (int j = 0; j < theta.length; j++) {
                sum += x[i][j] * theta[j];
            }
            J += (y[i] * Math.log(sigmoid(sum))) + ((1.0 - y[i]) * Math.log(1.0 - sigmoid(sum)));
        }
        return -J / m;
    }

    private static double[] grad(double[] theta, double[][] x, double[] y) {
        double[] grad = new double[theta.length];
        int m = y.length;
        double[] temp = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            double sum = 0;
            for (int j = 0; j < theta.length; j++) {
                sum += x[i][j] * theta[j];
            }
            temp[i] = (sigmoid(sum) - y[i]) / m;
        }
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < theta.length; j++) {
                grad[j] += x[i][j] * temp[i];
            }
        }
        return grad;
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static void addFaces(LinkedList<Mat> faces) {
        faces.forEach((face) -> {
            int colCounter = 0;
            for (int i = 0; i < face.rows(); i++) {
                for (int j = 0; j < face.cols(); j++) {
                    X[counter][colCounter++] = face.get(i, j)[0];
                }
            }
            counter++;
        });
    }

    public static void addFaces(double[][] faces) {
        for (int i = 0; i < faces.length; i++) {
            X[i] = faces[i];
        }
    }

}
