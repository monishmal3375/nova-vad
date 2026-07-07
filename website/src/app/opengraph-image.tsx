import { ImageResponse } from "next/og";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "NOVA-VAD — Voice activity detection that works in real noise";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "#05070c",
          backgroundImage:
            "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(76,211,245,0.16), transparent)",
          fontFamily: "sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            marginBottom: 36,
          }}
        >
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: "50%",
              background: "#0b0f18",
              border: "2px solid #4cd3f5",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 26,
              fontWeight: 700,
              color: "#4cd3f5",
            }}
          >
            N
          </div>
          <div style={{ display: "flex", fontSize: 34, fontWeight: 600, color: "#f1f5f9" }}>
            NOVA<span style={{ color: "#4cd3f5" }}>-VAD</span>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            fontSize: 56,
            fontWeight: 600,
            color: "#f1f5f9",
            textAlign: "center",
            lineHeight: 1.15,
            maxWidth: 920,
          }}
        >
          Voice activity detection that{" "}
          <span style={{ color: "#4cd3f5", marginLeft: 14 }}>works in real noise</span>
        </div>

        <div
          style={{
            display: "flex",
            gap: 14,
            marginTop: 48,
          }}
        >
          {[
            { value: "99.8%", label: "Accuracy" },
            { value: "MIT", label: "Licensed" },
            { value: "0", label: "GPUs required" },
          ].map((stat) => (
            <div
              key={stat.label}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                background: "#0b0f18",
                border: "1px solid #1e2738",
                borderRadius: 14,
                padding: "16px 28px",
              }}
            >
              <div style={{ display: "flex", fontSize: 28, fontWeight: 700, color: "#4cd3f5" }}>
                {stat.value}
              </div>
              <div style={{ display: "flex", fontSize: 14, color: "#8b96ac", marginTop: 4 }}>
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    ),
    { ...size }
  );
}
