import { NextResponse } from "next/server"

const REQUEST_TIMEOUT_MS = 45_000

function getBackendChatUrl(): string | null {
  const fullUrl = process.env.BACKEND_CHAT_URL?.trim()
  if (fullUrl) {
    return fullUrl
  }

  const baseUrl = process.env.BACKEND_BASE_URL?.trim()
  if (!baseUrl) {
    return null
  }

  return `${baseUrl.replace(/\/$/, "")}/api/chat`
}

function toErrorResponse(message: string, status: number): NextResponse {
  return NextResponse.json({ error: message }, { status })
}

export async function POST(request: Request): Promise<Response> {
  const backendChatUrl = getBackendChatUrl()
  if (!backendChatUrl) {
    return toErrorResponse(
      "Servern saknar backend-konfiguration. Sätt BACKEND_CHAT_URL eller BACKEND_BASE_URL.",
      500
    )
  }

  let requestBody: unknown
  try {
    requestBody = await request.json()
  } catch {
    return toErrorResponse("Ogiltig JSON i förfrågan.", 400)
  }

  const abortController = new AbortController()
  const timeoutId = setTimeout(() => abortController.abort(), REQUEST_TIMEOUT_MS)

  try {
    const upstreamResponse = await fetch(backendChatUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: request.headers.get("accept") ?? "text/event-stream",
      },
      body: JSON.stringify(requestBody),
      signal: abortController.signal,
      cache: "no-store",
    })

    if (!upstreamResponse.ok) {
      const errorBody = await upstreamResponse.text()
      const hasBody = typeof errorBody === "string" && errorBody.trim().length > 0
      return NextResponse.json(
        {
          error: hasBody
            ? `Chat-backend svarade med fel: ${errorBody}`
            : "Chat-backend svarade med ett fel.",
        },
        { status: upstreamResponse.status }
      )
    }

    if (!upstreamResponse.body) {
      return toErrorResponse("Chat-backend returnerade inget svar.", 502)
    }

    const responseHeaders = new Headers()
    const contentType = upstreamResponse.headers.get("content-type")
    if (contentType) {
      responseHeaders.set("content-type", contentType)
    }

    const cacheControl = upstreamResponse.headers.get("cache-control")
    responseHeaders.set("cache-control", cacheControl ?? "no-store")

    const streamHeaderNames = [
      "x-vercel-ai-data-stream",
      "x-vercel-ai-ui-message-stream",
      "x-vercel-ai-stream-protocol",
    ]

    for (const headerName of streamHeaderNames) {
      const headerValue = upstreamResponse.headers.get(headerName)
      if (headerValue) {
        responseHeaders.set(headerName, headerValue)
      }
    }

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: responseHeaders,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return toErrorResponse(
        "Tidsgränsen för chat-svaret överskreds. Försök igen om en stund.",
        504
      )
    }

    return toErrorResponse("Kunde inte nå chat-backend. Kontrollera serveranslutningen.", 502)
  } finally {
    clearTimeout(timeoutId)
  }
}
