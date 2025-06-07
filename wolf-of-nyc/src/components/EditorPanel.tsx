
import React, { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, Save, Upload, FileText, Book, Volume2, Image, FileAudio, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { createWorker } from 'tesseract.js';
import { VoiceSelector } from "@/components/VoiceSelector";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";

interface EditorPanelProps {
  dyslexicMode: boolean;
}

const EditorPanel = ({ dyslexicMode }: EditorPanelProps) => {
  const [text, setText] = useState<string>("Enter your script here...");
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentWordIndex, setCurrentWordIndex] = useState<number>(-1);
  const [playbackRate, setPlaybackRate] = useState<number>(1);
  const [volume, setVolume] = useState<number>(1);
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const [isProcessingImage, setIsProcessingImage] = useState<boolean>(false);
  const wordsRef = useRef<string[]>([]);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const speechSynthRef = useRef<SpeechSynthesisUtterance | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    wordsRef.current = e.target.value.split(/\s+/);
  };

  const togglePlayback = () => {
    if (isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  };

  // Load available voices
  useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        setAvailableVoices(voices);
        // Set default voice
        setSelectedVoice(voices[0]);
      }
    };
    
    loadVoices();
    
    // Chrome loads voices asynchronously
    if ('onvoiceschanged' in window.speechSynthesis) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
    
    return () => {
      if ('onvoiceschanged' in window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = null;
      }
    };
  }, []);

  const startPlayback = () => {
    // Check if browser supports speech synthesis
    if (!('speechSynthesis' in window)) {
      toast({
        title: "Speech synthesis not supported",
        description: "Your browser doesn't support text-to-speech functionality.",
        variant: "destructive"
      });
      return;
    }
    
    setIsPlaying(true);
    setCurrentWordIndex(0);
    
    // Cancel any previous speech
    window.speechSynthesis.cancel();
    
    // Create a new utterance for the current text
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = playbackRate;
    utterance.volume = volume;
    
    // Set the selected voice if available
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }
    
    // Store the utterance reference
    speechSynthRef.current = utterance;
    
    // Set up event handlers
    utterance.onstart = () => {
      setIsPlaying(true);
    };
    
    utterance.onend = () => {
      setIsPlaying(false);
      setCurrentWordIndex(-1);
    };
    
    utterance.onpause = () => {
      setIsPlaying(false);
    };
    
    utterance.onresume = () => {
      setIsPlaying(true);
    };
    
    utterance.onboundary = (event) => {
      // This event fires whenever a word or sentence boundary is reached
      if (event.name === 'word') {
        // Calculate which word we're on based on the character index
        const textUpToChar = text.substring(0, event.charIndex);
        const wordCount = textUpToChar.split(/\s+/).length - 1;
        setCurrentWordIndex(wordCount);
      }
    };
    
    // Start speaking
    window.speechSynthesis.speak(utterance);
  };

  const stopPlayback = () => {
    setIsPlaying(false);
    setCurrentWordIndex(-1);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    // Stop any active speech
    if (speechSynthRef.current) {
      window.speechSynthesis.cancel();
    }
  };

  // OCR functionality
  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setIsProcessingImage(true);
    toast({
      title: "Processing image",
      description: "Reading text from image...",
    });
    
    try {
      const worker = await createWorker();
      await worker.loadLanguage('eng');
      await worker.initialize('eng');
      const { data } = await worker.recognize(file);
      await worker.terminate();
      
      // Update the text area with OCR results
      setText(data.text);
      wordsRef.current = data.text.split(/\s+/);
      
      toast({
        title: "OCR complete",
        description: "Text extracted from image successfully.",
      });
    } catch (error) {
      toast({
        title: "OCR failed",
        description: "Failed to extract text from image.",
        variant: "destructive",
      });
      console.error("OCR error:", error);
    } finally {
      setIsProcessingImage(false);
      // Clear the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Export audio function
  const handleExportAudio = () => {
    // Since Web Speech API doesn't provide direct export,
    // inform the user this is a simulation
    toast({
      title: "Exporting audio",
      description: "Preparing audio export...",
    });
    
    // Create a simple text file to simulate export
    const element = document.createElement('a');
    const file = new Blob([text], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = 'script-export.txt';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    
    toast({
      title: "Export complete",
      description: "Script exported as text file. Audio export simulation complete.",
    });
  };

  // Export text function
  const handleExportText = () => {
    const element = document.createElement('a');
    const file = new Blob([text], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = 'script.txt';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    
    toast({
      title: "Export complete",
      description: "Script exported as text file.",
    });
  };

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      window.speechSynthesis.cancel();
    };
  }, []);

  // Update speech parameters when they change
  useEffect(() => {
    if (speechSynthRef.current && isPlaying) {
      // We need to restart speech with new parameters
      const currentIndex = currentWordIndex;
      stopPlayback();
      
      // Small timeout to ensure previous speech is fully canceled
      setTimeout(() => {
        setCurrentWordIndex(currentIndex);
        startPlayback();
      }, 100);
    }
  }, [playbackRate, volume, selectedVoice]);

  // Initialize words on component mount
  useEffect(() => {
    wordsRef.current = text.split(/\s+/);
  }, []);

  // Render text with highlighted current word
  const renderText = () => {
    if (currentWordIndex === -1) return text;
    
    const words = text.split(/\s+/);
    return words.map((word, index) => (
      <span 
        key={index} 
        className={index === currentWordIndex ? "text-scriptGold font-bold" : ""}
      >
        {word}{' '}
      </span>
    ));
  };

  return (
    <ResizablePanelGroup
      direction="vertical"
      className="h-full w-full rounded-lg border border-scriptRed/20"
    >
      <ResizablePanel defaultSize={80} minSize={30}>
        <div className="h-full">
          {isPlaying ? (
            <div 
              className={`editor-container h-full w-full p-4 overflow-y-auto text-white ${dyslexicMode ? 'font-dyslexic' : 'font-inter'}`}
            >
              {renderText()}
            </div>
          ) : (
            <textarea
              className={`editor-container h-full w-full p-4 bg-black text-white resize-none outline-none border-none ${dyslexicMode ? 'font-dyslexic' : 'font-inter'}`}
              value={text}
              onChange={handleTextChange}
              placeholder="Enter your script here..."
            />
          )}
        </div>
      </ResizablePanel>
      
      <ResizableHandle withHandle className="bg-scriptRed/20 border-y border-scriptRed/30" />
      
      <ResizablePanel defaultSize={20} minSize={15}>
        <div className="flex flex-col space-y-3 p-3 h-full">
          <div className="flex items-center justify-between">
            <div className="flex space-x-2">
              <Button 
                onClick={togglePlayback}
                variant="default"
                size="icon"
                className="bg-scriptRed hover:bg-scriptRed/80"
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              
              {/* Voice selection */}
              <Sheet>
                <SheetTrigger asChild>
                  <Button 
                    variant="outline" 
                    size="icon"
                    className="border-scriptRed/50 text-white"
                  >
                    <Volume2 className="h-4 w-4" />
                  </Button>
                </SheetTrigger>
                <SheetContent className="bg-gray-900 border-scriptRed/20">
                  <VoiceSelector
                    availableVoices={availableVoices}
                    selectedVoice={selectedVoice}
                    onVoiceSelect={setSelectedVoice}
                  />
                </SheetContent>
              </Sheet>
              
              {/* OCR button */}
              <Button 
                variant="outline" 
                size="icon"
                className="border-scriptRed/50 text-white"
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessingImage}
              >
                <Image className="h-4 w-4" />
              </Button>
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleImageUpload}
                accept="image/*"
                style={{ display: 'none' }}
              />
              
              {/* Export options */}
              <Button 
                variant="outline" 
                size="icon"
                className="border-scriptRed/50 text-white"
                onClick={handleExportText}
              >
                <FileText className="h-4 w-4" />
              </Button>
              
              <Button 
                variant="outline"
                size="icon" 
                className="border-scriptRed/50 text-white"
                onClick={handleExportAudio}
              >
                <FileAudio className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="flex items-center space-x-2">
              <FileText className="h-4 w-4 text-scriptRed" />
              <span className="text-sm text-white">Words: {wordsRef.current.length}</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <Book className="h-4 w-4 text-scriptRed" />
            <span className="text-sm text-white">Speed:</span>
            <Slider
              value={[playbackRate]}
              min={0.5}
              max={2}
              step={0.1}
              onValueChange={(values) => setPlaybackRate(values[0])}
              className="w-32"
            />
            <span className="text-sm text-white">{playbackRate.toFixed(1)}x</span>
          </div>
          
          <div className="flex items-center space-x-4">
            <Volume2 className="h-4 w-4 text-scriptRed" />
            <span className="text-sm text-white">Volume:</span>
            <Slider
              value={[volume]}
              min={0}
              max={1}
              step={0.1}
              onValueChange={(values) => setVolume(values[0])}
              className="w-32"
            />
            <span className="text-sm text-white">{Math.round(volume * 100)}%</span>
          </div>
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
};

export default EditorPanel;
