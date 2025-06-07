
export interface Voice {
  id: string;
  name: string;
  gender: 'male' | 'female';
  language: string;
}

export interface Project {
  id: string;
  name: string;
  content: string;
  lastEdited: string;
  voiceId?: string;
  settings: {
    speed: number;
    pitch: number;
    volume: number;
  };
}

export interface ExportOptions {
  format: 'mp3' | 'wav' | 'ogg';
  quality: 'low' | 'medium' | 'high';
}
