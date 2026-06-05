import { NextRequest, NextResponse } from "next/server";

// Disable all caching for this route — chat responses are always dynamic
export const dynamic = "force-dynamic";
export const fetchCache = "force-no-store";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch("http://127.0.0.1:8080/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to connect to backend" },
      { status: 500 }
    );
  }
}
