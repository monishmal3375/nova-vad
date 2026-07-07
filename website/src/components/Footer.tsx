import { Logo } from "./Logo";

export function Footer() {
  return (
    <footer className="border-t border-border px-6 py-10">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-6 sm:flex-row">
        <div className="flex items-center gap-2.5">
          <Logo size={24} animated={false} />
          <span className="font-display text-sm font-semibold tracking-tight text-foreground">
            NOVA<span className="text-primary">-VAD</span>
          </span>
        </div>

        <p className="text-xs text-muted">MIT License &middot; free to use, modify, and distribute</p>

        <div className="flex items-center gap-2 text-sm text-muted">
          <a href="https://github.com/monishmal3375/nova-vad" target="_blank" rel="noreferrer" className="inline-flex min-h-11 items-center px-2 transition-colors duration-200 hover:text-primary">
            GitHub
          </a>
          <a href="https://huggingface.co/monishmal0204/nova-vad" target="_blank" rel="noreferrer" className="inline-flex min-h-11 items-center px-2 transition-colors duration-200 hover:text-primary">
            Hugging Face
          </a>
          <a href="https://x.com/Nova_vad" target="_blank" rel="noreferrer" className="inline-flex min-h-11 items-center px-2 transition-colors duration-200 hover:text-primary">
            X
          </a>
        </div>
      </div>
    </footer>
  );
}
