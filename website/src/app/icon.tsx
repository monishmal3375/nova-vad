import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "50%",
          background: "#0b0f18",
          border: "1.5px solid #4cd3f5",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            fontSize: 15,
            fontWeight: 700,
            color: "#4cd3f5",
            fontFamily: "sans-serif",
          }}
        >
          N
        </div>
      </div>
    ),
    { ...size }
  );
}
