
import React, { useState } from 'react';
import ProjectSidebar from '@/components/ProjectSidebar';
import EditorPanel from '@/components/EditorPanel';

const Index = () => {
  const [dyslexicMode, setDyslexicMode] = useState<boolean>(false);

  return (
    <div className="min-h-screen flex flex-col bg-black text-white">
      <header className="bg-black border-b border-scriptRed/20 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <span className="text-scriptRed text-xl font-bold">Script</span>
            <span className="text-scriptGold text-xl font-bold">Voice</span>
          </div>
          <div className="text-sm text-gray-400">
            TTS Script Editor
          </div>
        </div>
      </header>
      
      <main className="flex-1 flex">
        <ProjectSidebar 
          dyslexicMode={dyslexicMode} 
          setDyslexicMode={setDyslexicMode} 
        />
        <div className="flex-1 flex flex-col">
          <div className="flex-1 p-4">
            <EditorPanel dyslexicMode={dyslexicMode} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
