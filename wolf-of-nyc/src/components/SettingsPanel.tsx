
import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Settings, Book, Volume2 } from "lucide-react";

interface SettingsPanelProps {
  dyslexicMode: boolean;
  setDyslexicMode: (value: boolean) => void;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ dyslexicMode, setDyslexicMode }) => {
  return (
    <div className="w-full bg-gray-900 border-t border-scriptRed/20 p-4">
      <Tabs defaultValue="voice">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium text-white flex items-center">
            <Settings className="h-4 w-4 mr-2 text-scriptRed" /> Settings
          </h3>
          <TabsList className="bg-black">
            <TabsTrigger value="voice" className="data-[state=active]:bg-scriptRed">Voice</TabsTrigger>
            <TabsTrigger value="accessibility" className="data-[state=active]:bg-scriptRed">Accessibility</TabsTrigger>
            <TabsTrigger value="export" className="data-[state=active]:bg-scriptRed">Export</TabsTrigger>
          </TabsList>
        </div>
        
        <TabsContent value="voice" className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="voice" className="text-white">Voice</Label>
              <Select defaultValue="male1">
                <SelectTrigger className="bg-black border-scriptRed/20">
                  <SelectValue placeholder="Select voice" />
                </SelectTrigger>
                <SelectContent className="bg-black border-scriptRed/20">
                  <SelectItem value="male1">Male Voice 1</SelectItem>
                  <SelectItem value="male2">Male Voice 2</SelectItem>
                  <SelectItem value="female1">Female Voice 1</SelectItem>
                  <SelectItem value="female2">Female Voice 2</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="language" className="text-white">Language</Label>
              <Select defaultValue="en-US">
                <SelectTrigger className="bg-black border-scriptRed/20">
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent className="bg-black border-scriptRed/20">
                  <SelectItem value="en-US">English (US)</SelectItem>
                  <SelectItem value="en-UK">English (UK)</SelectItem>
                  <SelectItem value="es">Spanish</SelectItem>
                  <SelectItem value="fr">French</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-white flex items-center">
                <Volume2 className="h-4 w-4 mr-2 text-scriptRed" /> Volume
              </Label>
              <span className="text-sm text-white">80%</span>
            </div>
            <Slider defaultValue={[80]} max={100} step={1} className="w-full" />
          </div>
          
          <div className="space-y-2">
            <Label className="text-white flex items-center">
              <Book className="h-4 w-4 mr-2 text-scriptRed" /> Pitch
            </Label>
            <Slider defaultValue={[50]} max={100} step={1} className="w-full" />
          </div>
        </TabsContent>
        
        <TabsContent value="accessibility" className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="dyslexic-mode" className="text-white">
              Dyslexia-Friendly Font
            </Label>
            <Switch 
              id="dyslexic-mode" 
              checked={dyslexicMode}
              onCheckedChange={setDyslexicMode}
              className="data-[state=checked]:bg-scriptRed"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="large-text" className="text-white">
              Large Text
            </Label>
            <Switch 
              id="large-text" 
              className="data-[state=checked]:bg-scriptRed"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="high-contrast" className="text-white">
              High Contrast
            </Label>
            <Switch 
              id="high-contrast" 
              className="data-[state=checked]:bg-scriptRed"
            />
          </div>
        </TabsContent>
        
        <TabsContent value="export" className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="export-format" className="text-white">Audio Format</Label>
            <Select defaultValue="mp3">
              <SelectTrigger className="bg-black border-scriptRed/20">
                <SelectValue placeholder="Select format" />
              </SelectTrigger>
              <SelectContent className="bg-black border-scriptRed/20">
                <SelectItem value="mp3">MP3</SelectItem>
                <SelectItem value="wav">WAV</SelectItem>
                <SelectItem value="ogg">OGG</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="quality" className="text-white">Quality</Label>
            <Select defaultValue="high">
              <SelectTrigger className="bg-black border-scriptRed/20">
                <SelectValue placeholder="Select quality" />
              </SelectTrigger>
              <SelectContent className="bg-black border-scriptRed/20">
                <SelectItem value="low">Low (faster)</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="high">High (slower)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SettingsPanel;
