
import React, { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { AlertTriangle, Camera, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { mediaPipeService } from '@/services/MediaPipeService';
import { api } from '@/services/api';

interface WebcamFeedProps {
  onTranslationResult: (text: string) => void;
}

const WebcamFeed: React.FC<WebcamFeedProps> = ({ onTranslationResult }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const handleLandmarks = async (landmarks: any) => {
    try {
      // Send landmarks to backend API for prediction
      const response = await api.translateHandLandmarks(landmarks);
      
      // Use the prediction result
      if (response && response.prediction) {
        onTranslationResult(response.prediction);
      }
    } catch (error) {
      console.error('Translation error:', error);
      
      if (error instanceof Error) {
        toast.error(`Translation failed: ${error.message}`);
      } else {
        toast.error('Translation failed. Please try again.');
      }
    }
  };

  const startTranslation = async () => {
    if (!videoRef.current) return;
    
    try {
      const success = await mediaPipeService.start(handleLandmarks);
      if (success) {
        setIsRunning(true);
        toast.success('Translation started');
      } else {
        throw new Error('Failed to start translation');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start translation';
      setError(message);
      toast.error(message);
    }
  };

  const stopTranslation = () => {
    mediaPipeService.stop();
    setIsRunning(false);
    toast.info('Translation stopped');
  };

  useEffect(() => {
    const initializeMediaPipe = async () => {
      try {
        setIsInitializing(true);
        setError(null);
        
        // Check if camera is available
        const isCameraAvailable = await mediaPipeService.isCameraAvailable();
        if (!isCameraAvailable) {
          throw new Error('No camera detected on your device');
        }
        
        // Initialize MediaPipe with the video element
        if (videoRef.current) {
          const initialized = await mediaPipeService.initialize(videoRef.current);
          if (!initialized) {
            throw new Error('Failed to initialize hand tracking');
          }
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to initialize';
        setError(message);
        toast.error(message);
      } finally {
        setIsInitializing(false);
      }
    };
    
    initializeMediaPipe();
    
    // Cleanup on unmount
    return () => {
      mediaPipeService.stop();
    };
  }, []);

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0 relative">
        {error ? (
          <div className="aspect-video bg-muted flex flex-col items-center justify-center p-6 text-center">
            <AlertTriangle className="h-12 w-12 text-destructive mb-4" />
            <h3 className="text-lg font-medium mb-2">Camera Error</h3>
            <p className="text-muted-foreground mb-4">{error}</p>
            <Button onClick={() => window.location.reload()}>
              Reload Page
            </Button>
          </div>
        ) : (
          <>
            <video 
              ref={videoRef}
              className="w-full aspect-video bg-black"
              autoPlay
              playsInline
              muted
            />
            
            {isInitializing && (
              <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center">
                <div className="flex flex-col items-center">
                  <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                  <p className="text-lg font-medium">Initializing Camera...</p>
                </div>
              </div>
            )}
            
            <div className="p-4 flex justify-center">
              {isRunning ? (
                <Button 
                  variant="destructive" 
                  onClick={stopTranslation}
                  disabled={isInitializing}
                >
                  Stop Translation
                </Button>
              ) : (
                <Button 
                  onClick={startTranslation}
                  disabled={isInitializing || !!error}
                  className="gap-2"
                >
                  <Camera className="h-4 w-4" /> Start Translation
                </Button>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default WebcamFeed;
