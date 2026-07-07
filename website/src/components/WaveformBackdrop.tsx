export function WaveformBackdrop() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 overflow-hidden grid-fade-mask"
    >
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-10%,rgba(76,211,245,0.14),transparent)]" />
      <svg
        className="absolute left-1/2 top-16 h-[420px] w-[1400px] -translate-x-1/2 opacity-[0.35] sm:top-20"
        viewBox="0 0 1400 420"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M0 210 C 60 90, 120 90, 180 210 C 240 330, 300 90, 360 210 C 420 330, 480 90, 540 210 C 600 330, 660 90, 720 210 C 780 330, 840 90, 900 210 C 960 330, 1020 90, 1080 210 C 1140 330, 1200 90, 1260 210 C 1320 330, 1360 210, 1400 210"
          stroke="var(--wave-blue)"
          strokeWidth="1.5"
          className="animate-wave-drift"
        />
        <path
          d="M0 230 C 70 340, 130 340, 190 230 C 250 120, 310 340, 370 230 C 430 120, 490 340, 550 230 C 610 120, 670 340, 730 230 C 790 120, 850 340, 910 230 C 970 120, 1030 340, 1090 230 C 1150 120, 1210 340, 1270 230 C 1330 120, 1370 230, 1400 230"
          stroke="var(--primary)"
          strokeWidth="1.5"
          className="animate-wave-drift-slow"
        />
      </svg>
    </div>
  );
}
