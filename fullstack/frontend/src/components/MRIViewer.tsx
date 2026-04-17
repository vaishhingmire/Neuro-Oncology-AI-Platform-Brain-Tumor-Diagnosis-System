import { useState } from 'react';

interface MRIViewerProps {
    images: {
        original: string;
        detection: string;
        gradcam: string;
        segmentation: string;
    };
}

export function MRIViewer({ images }: MRIViewerProps) {
    const [opacity, setOpacity] = useState({
        detection: 100,
        gradcam: 50,
        segmentation: 100,
    });

    const [activeView, setActiveView] = useState<'all' | 'original' | 'gradcam' | 'segmentation'>('all');

    return (
        <div className="flex flex-col space-y-4">

            {/* Opacity Controls */}
            <div className="grid grid-cols-3 gap-4 mb-2">
                {Object.entries(opacity).map(([key, val]) => (
                    <div key={key} className="flex flex-col space-y-1">
                        <label className="text-xs font-semibold text-neutral-400 uppercase">
                            {key} Opacity
                        </label>
                        <input
                            type="range"
                            className="accent-blue-500"
                            min="0" max="100"
                            value={val}
                            onChange={(e) => setOpacity({ ...opacity, [key]: parseInt(e.target.value) })}
                        />
                    </div>
                ))}
            </div>

            {/* View Toggle Buttons */}
            <div className="flex space-x-2">
                {(['all', 'original', 'gradcam', 'segmentation'] as const).map((view) => (
                    <button
                        key={view}
                        onClick={() => setActiveView(view)}
                        className={`px-3 py-1 rounded text-xs font-semibold uppercase transition-colors ${activeView === view
                                ? 'bg-blue-600 text-white'
                                : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
                            }`}
                    >
                        {view}
                    </button>
                ))}
            </div>

            {/* Image Viewer */}
            <div className="relative w-full bg-black rounded-lg overflow-hidden flex items-center justify-center group"
                style={{ minHeight: '400px' }}>

                {/* Original — always shown as base */}
                <img
                    src={images.original}
                    className="absolute max-h-full max-w-full object-contain"
                    alt="Original MRI"
                />

                {/* Detection overlay */}
                {(activeView === 'all') && (
                    <img
                        src={images.detection}
                        className="absolute max-h-full max-w-full object-contain transition-opacity duration-200 pointer-events-none"
                        style={{ opacity: opacity.detection / 100 }}
                        alt="Detection"
                    />
                )}

                {/* GradCAM overlay */}
                {(activeView === 'all' || activeView === 'gradcam') && (
                    <img
                        src={images.gradcam}
                        className="absolute max-h-full max-w-full object-contain mix-blend-screen transition-opacity duration-200 pointer-events-none"
                        style={{ opacity: opacity.gradcam / 100 }}
                        alt="GradCAM"
                    />
                )}

                {/* Segmentation overlay */}
                {(activeView === 'all' || activeView === 'segmentation') && (
                    <img
                        src={images.segmentation}
                        className="absolute max-h-full max-w-full object-contain transition-opacity duration-200 pointer-events-none"
                        style={{ opacity: opacity.segmentation / 100 }}
                        alt="Segmentation"
                    />
                )}

                {/* ✅ FIXED: Crosshair ONLY shows on hover — was always visible before */}
                <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"
                    style={{
                        background: [
                            'linear-gradient(transparent 49.5%, rgba(0,255,0,0.3) 49.5%, rgba(0,255,0,0.3) 50.5%, transparent 50.5%)',
                            'linear-gradient(90deg, transparent 49.5%, rgba(0,255,0,0.3) 49.5%, rgba(0,255,0,0.3) 50.5%, transparent 50.5%)'
                        ].join(', ')
                    }}
                />

                {/* View label badge */}
                <div className="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded uppercase font-semibold">
                    {activeView === 'all' ? 'Multi-Model Overlay' : activeView}
                </div>
            </div>
        </div>
    );
}