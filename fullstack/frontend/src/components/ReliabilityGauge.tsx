interface ReliabilityGaugeProps {
    // FIX: backend returns reliability_percentage (0–100), not a 0–1 float
    score: number;  // 0–100
    tier: string;
}

export function ReliabilityGauge({ score, tier }: ReliabilityGaugeProps) {
    // Clamp to [0, 100] then render as percentage
    const percentage = Math.round(Math.max(0, Math.min(100, score)));
    // Normalise to 0–1 for SVG stroke calculation
    const fraction = percentage / 100;

    let color = "text-green-500";
    let ring = "stroke-green-500";

    if (tier === "Medium") {
        color = "text-yellow-500";
        ring = "stroke-yellow-500";
    } else if (tier === "Low") {
        color = "text-red-500";
        ring = "stroke-red-500";
    }

    const radius = 30;
    const circumference = 2 * Math.PI * radius;
    // FIX: use fraction (0–1) for stroke math, not the raw score
    const strokeDashoffset = circumference - (fraction * circumference);

    return (
        <div className="flex items-center space-x-6">
            <div className="relative w-24 h-24">
                <svg className="w-full h-full transform -rotate-90">
                    {/* Background circle */}
                    <circle
                        cx="48"
                        cy="48"
                        r={radius}
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="transparent"
                        className="text-neutral-700"
                    />
                    {/* Progress circle */}
                    <circle
                        cx="48"
                        cy="48"
                        r={radius}
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="transparent"
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeDashoffset}
                        className={`${ring} transition-all duration-1000 ease-out`}
                        strokeLinecap="round"
                    />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center flex-col">
                    <span className="text-xl font-bold text-white">{percentage}%</span>
                </div>
            </div>
            <div>
                <div className={`text-2xl font-bold ${color}`}>{tier} Trust</div>
                <p className="text-neutral-400 text-sm mt-1">Multi-model consensus</p>
            </div>
        </div>
    );
}
