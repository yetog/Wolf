
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import { Folder, FolderOpen, FileText, Plus, Search, Edit, Save, Settings } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Book, Volume2 } from "lucide-react";

interface Project {
  id: string;
  name: string;
  lastEdited: string;
  notes?: string;
}

interface ProjectSidebarProps {
  dyslexicMode: boolean;
  setDyslexicMode: (value: boolean) => void;
}

const ProjectSidebar: React.FC<ProjectSidebarProps> = ({ dyslexicMode, setDyslexicMode }) => {
  const [projects, setProjects] = useState<Project[]>([
    { 
      id: '1', 
      name: 'Welcome Script', 
      lastEdited: '2025-05-21',
      notes: 'Introduction script for new users.' 
    },
    { 
      id: '2', 
      name: 'Product Demo', 
      lastEdited: '2025-05-20',
      notes: 'Showcase of product features.' 
    },
    { 
      id: '3', 
      name: 'Podcast Intro', 
      lastEdited: '2025-05-19',
      notes: 'Opening for weekly podcast episode.' 
    },
  ]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedProject, setSelectedProject] = useState('1');
  const [editingNotes, setEditingNotes] = useState(false);
  const [currentNotes, setCurrentNotes] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);

  const filteredProjects = projects.filter(project => 
    project.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const addNewProject = () => {
    const newProject = {
      id: Date.now().toString(),
      name: 'New Script',
      lastEdited: new Date().toISOString().split('T')[0],
      notes: '',
    };
    setProjects([...projects, newProject]);
    setSelectedProject(newProject.id);
  };

  const handleNotesEdit = () => {
    const currentProject = projects.find(p => p.id === selectedProject);
    if (currentProject) {
      setCurrentNotes(currentProject.notes || '');
      setEditingNotes(true);
    }
  };

  const saveNotes = () => {
    setProjects(projects.map(project => 
      project.id === selectedProject ? { ...project, notes: currentNotes } : project
    ));
    setEditingNotes(false);
  };

  const selectedProjectData = projects.find(p => p.id === selectedProject);

  const toggleSettings = () => {
    setSettingsOpen(!settingsOpen);
  };

  return (
    <div className="w-64 bg-black border-r border-scriptRed/20 h-full flex flex-col">
      <div className="p-4 border-b border-scriptRed/20">
        <h2 className="text-xl font-bold text-white mb-4">Projects</h2>
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-scriptRed" />
          <Input
            type="text"
            placeholder="Search scripts..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8 bg-gray-900 border-scriptRed/20 text-white"
          />
        </div>
      </div>
      
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-2">
          {filteredProjects.map((project) => (
            <div
              key={project.id}
              className={`p-2 rounded-md cursor-pointer flex items-center space-x-2 ${
                selectedProject === project.id ? 'bg-scriptRed/20' : 'hover:bg-gray-900'
              }`}
              onClick={() => setSelectedProject(project.id)}
            >
              {selectedProject === project.id ? (
                <FolderOpen className="h-4 w-4 text-scriptRed" />
              ) : (
                <Folder className="h-4 w-4 text-scriptRed" />
              )}
              <div className="flex-1 overflow-hidden">
                <p className="text-sm text-white truncate">{project.name}</p>
                <p className="text-xs text-gray-400">{project.lastEdited}</p>
              </div>
              <FileText className="h-3 w-3 text-gray-500" />
            </div>
          ))}
        </div>
      </ScrollArea>
      
      {/* Notes section */}
      <div className="p-4 border-t border-b border-scriptRed/20">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-sm font-semibold text-white">Notes</h3>
          {!editingNotes ? (
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={handleNotesEdit}
              className="h-6 w-6 text-scriptRed hover:bg-scriptRed/20"
            >
              <Edit className="h-3 w-3" />
            </Button>
          ) : (
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={saveNotes}
              className="h-6 w-6 text-scriptRed hover:bg-scriptRed/20"
            >
              <Save className="h-3 w-3" />
            </Button>
          )}
        </div>
        
        {editingNotes ? (
          <Textarea 
            value={currentNotes}
            onChange={(e) => setCurrentNotes(e.target.value)} 
            placeholder="Add notes about your script..."
            className="bg-gray-900 border-scriptRed/20 text-white text-xs h-24 resize-none"
          />
        ) : (
          <Card className="bg-gray-900 border-scriptRed/20 h-24 overflow-auto">
            <CardContent className="p-2 text-xs text-gray-300">
              {selectedProjectData?.notes || 'No notes for this script.'}
            </CardContent>
          </Card>
        )}
      </div>
      
      {/* Buttons section */}
      <div className="p-4 space-y-3">
        <Button 
          onClick={addNewProject}
          className="w-full bg-scriptRed hover:bg-scriptRed/80 text-white"
        >
          <Plus className="h-4 w-4 mr-1" /> New Script
        </Button>
        
        <Button 
          onClick={toggleSettings}
          variant={settingsOpen ? "default" : "outline"}
          className={`w-full ${settingsOpen ? "bg-scriptRed hover:bg-scriptRed/80" : "border-scriptRed/50"} text-white`}
        >
          <Settings className="h-4 w-4 mr-1" /> Settings
        </Button>
      </div>
      
      {/* Settings Panel - Now inside the sidebar */}
      {settingsOpen && (
        <div className="p-4 bg-gray-900 border-t border-scriptRed/20">
          <Tabs defaultValue="voice" className="w-full">
            <TabsList className="bg-black w-full grid grid-cols-3">
              <TabsTrigger value="voice" className="data-[state=active]:bg-scriptRed">Voice</TabsTrigger>
              <TabsTrigger value="accessibility" className="data-[state=active]:bg-scriptRed">Access</TabsTrigger>
              <TabsTrigger value="export" className="data-[state=active]:bg-scriptRed">Export</TabsTrigger>
            </TabsList>
            
            <TabsContent value="voice" className="space-y-3 mt-3">
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
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label className="text-white text-xs">Volume</Label>
                  <span className="text-xs text-white">80%</span>
                </div>
                <Slider defaultValue={[80]} max={100} step={1} className="w-full" />
              </div>
              
              <div className="space-y-1">
                <Label className="text-white text-xs">Pitch</Label>
                <Slider defaultValue={[50]} max={100} step={1} className="w-full" />
              </div>
            </TabsContent>
            
            <TabsContent value="accessibility" className="space-y-3 mt-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="dyslexic-mode" className="text-white text-xs">
                  Dyslexia Font
                </Label>
                <Switch 
                  id="dyslexic-mode" 
                  checked={dyslexicMode}
                  onCheckedChange={setDyslexicMode}
                  className="data-[state=checked]:bg-scriptRed"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="large-text" className="text-white text-xs">
                  Large Text
                </Label>
                <Switch 
                  id="large-text" 
                  className="data-[state=checked]:bg-scriptRed"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="high-contrast" className="text-white text-xs">
                  High Contrast
                </Label>
                <Switch 
                  id="high-contrast" 
                  className="data-[state=checked]:bg-scriptRed"
                />
              </div>
            </TabsContent>
            
            <TabsContent value="export" className="space-y-3 mt-3">
              <div className="space-y-2">
                <Label htmlFor="export-format" className="text-white">Format</Label>
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
      )}
    </div>
  );
};

export default ProjectSidebar;
