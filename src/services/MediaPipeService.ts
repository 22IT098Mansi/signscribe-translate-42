
import * as mediapipe from '@mediapipe/hands';

class MediaPipeService {
  private hands: mediapipe.Hands | null = null;
  private videoElement: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private isRunning = false;
  private landmarkCallback: ((landmarks: number[]) => void) | null = null;

  // Check if camera is available
  async isCameraAvailable(): Promise<boolean> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.some(device => device.kind === 'videoinput');
    } catch (error) {
      console.error('Error checking for camera:', error);
      return false;
    }
  }

  // Initialize MediaPipe Hands with the video element
  async initialize(videoElement: HTMLVideoElement): Promise<boolean> {
    try {
      this.videoElement = videoElement;
      
      // Initialize MediaPipe Hands
      this.hands = new mediapipe.Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });
      
      // Configure MediaPipe Hands
      await this.hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      
      // Set up result handling
      this.hands.onResults(results => {
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0 && this.landmarkCallback) {
          // Extract keypoints from landmarks
          const keypoints = this.extractKeypoints(results.multiHandLandmarks);
          this.landmarkCallback(keypoints);
        }
      });
      
      return true;
    } catch (error) {
      console.error('Error initializing MediaPipe Hands:', error);
      return false;
    }
  }

  // Start processing with callback for landmarks
  async start(callback: (landmarks: number[]) => void): Promise<boolean> {
    if (!this.videoElement || !this.hands) {
      console.error('MediaPipe not initialized');
      return false;
    }
    
    try {
      // Request camera access
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' }
      });
      
      // Set video source
      this.videoElement.srcObject = this.stream;
      
      // Set callback
      this.landmarkCallback = callback;
      
      // Start tracking
      const camera = new mediapipe.Camera(this.videoElement, {
        onFrame: async () => {
          if (this.isRunning && this.hands) {
            await this.hands.send({ image: this.videoElement! });
          }
        },
        width: 640,
        height: 480
      });
      camera.start();
      
      this.isRunning = true;
      return true;
    } catch (error) {
      console.error('Error starting MediaPipe:', error);
      return false;
    }
  }

  // Stop processing
  stop(): void {
    this.isRunning = false;
    
    // Stop stream tracks
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    
    // Clear callback
    this.landmarkCallback = null;
  }

  // Extract keypoints from landmarks in the format expected by the model
  private extractKeypoints(multiHandLandmarks: mediapipe.NormalizedLandmarkList[]): number[] {
    const keypoints: number[] = [];
    
    // Process up to 2 hands (left and right)
    if (multiHandLandmarks.length > 0) {
      for (let i = 0; i < Math.min(multiHandLandmarks.length, 2); i++) {
        const handLandmarks = multiHandLandmarks[i];
        
        // Flatten the landmarks (x, y, z for each of the 21 hand landmarks)
        for (const landmark of handLandmarks) {
          keypoints.push(landmark.x, landmark.y, landmark.z);
        }
      }
      
      // If only one hand is detected, pad with zeros for the second hand
      if (multiHandLandmarks.length === 1) {
        // Add zeros for the missing hand (21 landmarks × 3 coordinates)
        keypoints.push(...Array(21 * 3).fill(0));
      }
    } else {
      // No hands detected, fill with zeros for two hands
      keypoints.push(...Array(2 * 21 * 3).fill(0));
    }
    
    return keypoints;
  }
}

// Export singleton instance
export const mediaPipeService = new MediaPipeService();
