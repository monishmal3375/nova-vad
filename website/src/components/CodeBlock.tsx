"use client";

import { useState } from "react";

function highlight(line: string, key: number) {
  if (line.trim().startsWith("#")) {
    return (
      <div key={key} className="text-muted">
        {line}
      </div>
    );
  }
  const parts = line.split(" ");
  return (
    <div key={key}>
      {parts.map((part, i) => {
        const isCommand = i === 0 && !line.startsWith(" ");
        const isFlag = part.startsWith("-");
        const isPath = part.includes("/") || part.includes(".py") || part.includes(".wav");
        return (
          <span
            key={i}
            className={
              isCommand
                ? "text-primary"
                : isFlag
                  ? "text-wave-blue"
                  : isPath
                    ? "text-foreground"
                    : "text-foreground/90"
            }
          >
            {part}
            {i !== parts.length - 1 ? " " : ""}
          </span>
        );
      })}
    </div>
  );
}

export function CodeBlock({ code, label }: { code: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-danger/50" />
          <span className="h-2.5 w-2.5 rounded-full bg-primary/40" />
          <span className="h-2.5 w-2.5 rounded-full bg-wave-blue/50" />
          {label && <span className="ml-3 text-xs text-muted">{label}</span>}
        </div>
        <button
          onClick={onCopy}
          className="inline-flex min-h-11 cursor-pointer items-center rounded-md px-3 text-xs font-medium text-muted transition-colors duration-150 hover:text-primary"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-4 font-mono text-[13px] leading-relaxed">
        <code>{code.split("\n").map((line, i) => highlight(line, i))}</code>
      </pre>
    </div>
  );
}
