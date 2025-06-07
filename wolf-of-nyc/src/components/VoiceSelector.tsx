
import React from 'react';
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

interface VoiceSelectorProps {
  availableVoices: SpeechSynthesisVoice[];
  selectedVoice: SpeechSynthesisVoice | null;
  onVoiceSelect: (voice: SpeechSynthesisVoice) => void;
}

export const VoiceSelector: React.FC<VoiceSelectorProps> = ({ 
  availableVoices, 
  selectedVoice, 
  onVoiceSelect 
}) => {
  return (
    <div className="p-4 space-y-4">
      <h3 className="text-lg font-medium text-white">Select Voice</h3>
      <RadioGroup 
        value={selectedVoice?.voiceURI || ''} 
        onValueChange={(value) => {
          const voice = availableVoices.find(v => v.voiceURI === value);
          if (voice) {
            onVoiceSelect(voice);
          }
        }}
      >
        <div className="space-y-2 max-h-64 overflow-y-auto pr-2">
          {availableVoices.length === 0 ? (
            <div className="text-white/70">No voices available</div>
          ) : (
            availableVoices.map((voice) => (
              <div 
                key={voice.voiceURI} 
                className="flex items-center space-x-2 bg-gray-800/50 p-3 rounded-md"
              >
                <RadioGroupItem 
                  value={voice.voiceURI} 
                  id={voice.voiceURI} 
                  className="border-scriptRed text-scriptRed"
                />
                <Label 
                  htmlFor={voice.voiceURI} 
                  className="flex-1 cursor-pointer text-sm text-white"
                >
                  <div className="font-medium">{voice.name}</div>
                  <div className="text-xs text-white/70">
                    {voice.lang} • {voice.localService ? 'Local' : 'Remote'}
                  </div>
                </Label>
              </div>
            ))
          )}
        </div>
      </RadioGroup>
    </div>
  );
};
