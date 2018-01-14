import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Point;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.videoio.VideoCapture;

/* Make sure GRIP is not open when running
 * 
 * 
 */

public class Main {
	public static boolean windowOpen;

	// Camera FOV, change if switching cameras
	private static final int fovx = 60;
	private static final int fovy = 50;

	public static void main(String[] args) {
		// Loads OpenCV
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// Creates the window
		JFrame frame = new JFrame("Vision Window");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(800, 600);

		// Create a window listener that provides callback for opened and closed
		WindowListener windowListen = new WindowListener() {
			@Override
			public void windowActivated(WindowEvent arg0) {
			}

			@Override
			public void windowClosed(WindowEvent argo0) {
			}

			@Override
			public void windowClosing(WindowEvent arg0) {
				Main.windowOpen = false;
			}

			@Override
			public void windowDeactivated(WindowEvent arg0) {
			}

			@Override
			public void windowDeiconified(WindowEvent arg0) {
			}

			@Override
			public void windowIconified(WindowEvent arg0) {
			}

			@Override
			public void windowOpened(WindowEvent arg0) {
				Main.windowOpen = true;
			}
		};
		frame.addWindowListener(windowListen);

		// Creates video on window
		JLabel cameraVision = new JLabel();
		frame.add(cameraVision);
		frame.setVisible(true);

		// Adds GRIP pipeline
		GripPipeline pipeline = new GripPipeline();

		// Sets the camera, change cameraIndex to 0 for pi
		int cameraIndex = 1;
		VideoCapture camera = new VideoCapture(cameraIndex);

		// If camera isn't open
		while (!camera.isOpened()) {
			camera.open(cameraIndex);
			System.out.println("Camera didn't open");
		}

		// Stores image from camera
		Mat image = new Mat();
		camera.read(image);
		
		// Main loop while window is open
		while (Main.windowOpen) {
			
			// Updates video feed with new image
			cameraVision.setIcon(new ImageIcon(matToBufferedImage(image)));
			
			// Gives error if can't read image
			if (!camera.read(image)) {
				System.out.println("Can't read image");
				break;
			}
			
			// Processes image
			pipeline.process(image);
			
			// If no hulls, restart loop
			if (pipeline.convexHullsOutput().size() == 0) {
				continue;
			}
			
			// Finds the biggest hull
			MatOfPoint largestHull = pipeline.convexHullsOutput().get(0);
			for (int i = 0; i < pipeline.convexHullsOutput().size(); ++i) {
				if (Imgproc.contourArea(largestHull) < Imgproc.contourArea(pipeline.convexHullsOutput().get(i))) {
					largestHull = pipeline.convexHullsOutput().get(i);
				}
			}
			
			// Finds center of largestHull
			Point center = centerOfConvexHull(largestHull);

			// Draws circle on center
			Imgproc.circle(image, center, 10, new Scalar(255, 0, 0), 10);
			
			// Sets angleX and angleY
			double angleX = findAngle(center.x, image.cols(), fovx);
			double angleY = findAngle(center.y, image.rows(), fovy);

		}
		// Turns off camera
		System.out.println("Program done");
		camera.release();
	}
	
	// Convert OpenCV image to java image
	public static BufferedImage matToBufferedImage(Mat src) {
		BufferedImage image = new BufferedImage(src.cols(), src.rows(), BufferedImage.TYPE_3BYTE_BGR);

		src.get(0, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData());
		return image;
	}

	// Finds center of convex hull
	public static Point centerOfConvexHull(MatOfPoint hull) {
		Moments moment = Imgproc.moments(hull);
		Point center = new Point();
		center.x = (int) (moment.get_m10() / moment.get_m00());
		center.y = (int) (moment.get_m01() / moment.get_m00());
		return center;
	}

	//  Find angle from pixel data
	public static double findAngle(double pixel, int resolution, int fov) {
		double center = pixel - (resolution / 2);
		double fovtoradians = (Math.PI / 180) * fov;
		double ratio = center * (Math.sin(.5 * fovtoradians) / (.5 * resolution));
		double radians = Math.asin(ratio);
		double out = (180 / Math.PI) * radians;
		return out;
	}
}
