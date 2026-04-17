import { useRef, useState } from 'react';

interface UploadPanelProps {
    onUpload: (file: File) => void;
    loading: boolean;
    loadingStage?: string;
}

export function UploadPanel({ onUpload, loading, loadingStage }: UploadPanelProps) {
    const [dragging, setDragging] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const stages = [
        "Preprocessing scan...",
        "Running CNN classification...",
        "Computing Grad-CAM...",
        "Prompting SAM segmentation...",
        "Estimating volume...",
        "Finalizing diagnosis...",
    ];
    const stageIdx = stages.indexOf(loadingStage ?? "");
    const progress = loading ? Math.max(((stageIdx + 1) / stages.length) * 100, 10) : 0;

    const handleFile = (file: File) => {
        if (!file) return;
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG).');
            return;
        }
        onUpload(file);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault(); setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-[70vh] p-8">
            <div
                onClick={() => !loading && inputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                className="w-full max-w-lg rounded-2xl border-2 border-dashed p-12 text-center transition-all cursor-pointer"
                style={{
                    borderColor: dragging ? '#1d4ed8' : '#bfdbfe',
                    background: dragging ? '#eff6ff' : 'white',
                    boxShadow: dragging ? '0 0 0 4px rgba(29,78,216,0.1)' : '0 2px 16px rgba(30,58,138,0.07)',
                }}>

                <input ref={inputRef} type="file" accept="image/*" className="hidden"
                    onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />

                {loading ? (
                    <div className="space-y-5">
                        {/* Spinner */}
                        <div className="flex justify-center">
                            <div className="w-14 h-14 rounded-full border-4 border-t-transparent animate-spin"
                                style={{ borderColor: '#bfdbfe', borderTopColor: '#1d4ed8' }} />
                        </div>

                        {/* Stage label */}
                        <p className="text-sm font-semibold" style={{ color: '#1d4ed8' }}>
                            {loadingStage ?? "Analyzing..."}
                        </p>

                        {/* Progress bar */}
                        <div className="w-full rounded-full h-2.5" style={{ background: '#e0e7ff' }}>
                            <div className="h-2.5 rounded-full transition-all duration-700"
                                style={{ width: `${progress}%`, background: 'linear-gradient(90deg, #1d4ed8, #60a5fa)' }} />
                        </div>

                        {/* Stage dots */}
                        <div className="flex justify-center space-x-2">
                            {stages.map((_s, i) => (
                                <div key={i} className="w-2 h-2 rounded-full transition-all"
                                    style={{ background: i <= stageIdx ? '#1d4ed8' : '#bfdbfe' }} />
                            ))}
                        </div>

                        <p className="text-xs" style={{ color: '#94a3b8' }}>
                            Step {Math.max(stageIdx + 1, 1)} of {stages.length}
                        </p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {/* Icon */}
                        <div className="flex justify-center">
                            <div className="w-16 h-16 rounded-full flex items-center justify-center"
                                style={{ background: 'linear-gradient(135deg, #1e3a8a, #1d4ed8)' }}>
                                <svg width="28" height="28" fill="none" viewBox="0 0 24 24">
                                    <path d="M12 16V4m0 0L8 8m4-4 4 4" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="white" strokeWidth="2" strokeLinecap="round" />
                                </svg>
                            </div>
                        </div>

                        <div>
                            <p className="text-lg font-bold mb-1" style={{ color: '#1e3a8a' }}>
                                Initialize AI Radiomics Pipeline
                            </p>
                            <p className="text-sm" style={{ color: '#64748b' }}>
                                Drag and drop a patient MRI scan to begin synchronized<br />
                                CNN classification, Grad-CAM, SAM segmentation, and volume scoring.
                            </p>
                        </div>

                        <button className="px-6 py-2.5 rounded-xl text-sm font-semibold text-white transition-all"
                            style={{ background: 'linear-gradient(135deg, #1d4ed8, #3b82f6)' }}>
                            Select Scan File
                        </button>

                        <p className="text-xs" style={{ color: '#94a3b8' }}>Supports JPG, PNG · Any MRI modality</p>
                    </div>
                )}
            </div>
        </div>
    );
}