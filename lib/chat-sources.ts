import type { UIMessage } from "ai"

export type ChatSource = {
  title: string
  url: string
}

type UnknownRecord = Record<string, unknown>

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null
}

function toArray(value: unknown): unknown[] {
  if (Array.isArray(value)) {
    return value
  }

  return value == null ? [] : [value]
}

function getString(record: UnknownRecord, keys: string[]): string | null {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim()
    }
  }

  return null
}

function normalizeUrl(value: unknown): string | null {
  if (typeof value !== "string") {
    return null
  }

  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  try {
    const url = new URL(trimmed)
    if (url.protocol === "http:" || url.protocol === "https:") {
      return url.toString()
    }
  } catch {
    return null
  }

  return null
}

function titleFromUrl(url: string): string {
  try {
    const parsed = new URL(url)
    return parsed.hostname.replace(/^www\./, "")
  } catch {
    return "Källa"
  }
}

function sourceFromUnknown(value: unknown): ChatSource | null {
  if (!isRecord(value)) {
    return null
  }

  const rawUrl = getString(value, ["url", "href", "link", "source", "uri"])
  const normalizedUrl = normalizeUrl(rawUrl)
  if (!normalizedUrl) {
    return null
  }

  const rawTitle = getString(value, ["title", "label", "name", "text"])
  return {
    title: rawTitle ?? titleFromUrl(normalizedUrl),
    url: normalizedUrl,
  }
}

function sourceFromPart(part: UIMessage["parts"][number]): ChatSource | null {
  if (part.type === "source-url") {
    const normalizedUrl = normalizeUrl(part.url)
    if (!normalizedUrl) {
      return null
    }

    return {
      title: part.title?.trim() || titleFromUrl(normalizedUrl),
      url: normalizedUrl,
    }
  }

  if (part.type === "source-document") {
    const metadataUrl = isRecord(part.providerMetadata)
      ? getString(part.providerMetadata, ["url", "href", "link"])
      : null
    const normalizedUrl = normalizeUrl(metadataUrl)
    if (!normalizedUrl) {
      return null
    }

    return {
      title: part.title?.trim() || titleFromUrl(normalizedUrl),
      url: normalizedUrl,
    }
  }

  if (part.type.startsWith("data-") && "data" in part && isRecord(part.data)) {
    return sourceFromUnknown(part.data)
  }

  return null
}

function dedupeSources(sources: ChatSource[]): ChatSource[] {
  const seen = new Set<string>()
  const uniqueSources: ChatSource[] = []

  for (const source of sources) {
    const key = source.url.toLowerCase()
    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    uniqueSources.push(source)
  }

  return uniqueSources
}

export function extractSourcesFromMessage(message: UIMessage | Record<string, unknown>): ChatSource[] {
  const unknownMessage = message as Record<string, unknown>
  const sources: ChatSource[] = []

  const maybeParts = (unknownMessage as { parts?: unknown }).parts
  const parts = Array.isArray(maybeParts) ? (maybeParts as UIMessage["parts"]) : []

  for (const part of parts) {
    const source = sourceFromPart(part)
    if (source) {
      sources.push(source)
    }
  }

  const annotations = toArray(unknownMessage.annotations)
  for (const annotation of annotations) {
    if (!isRecord(annotation)) {
      continue
    }

    for (const candidate of toArray(annotation.sources)) {
      const source = sourceFromUnknown(candidate)
      if (source) {
        sources.push(source)
      }
    }

    for (const candidate of toArray(annotation.citations)) {
      const source = sourceFromUnknown(candidate)
      if (source) {
        sources.push(source)
      }
    }
  }

  for (const candidate of toArray(unknownMessage.sources)) {
    const source = sourceFromUnknown(candidate)
    if (source) {
      sources.push(source)
    }
  }

  for (const candidate of toArray(unknownMessage.citations)) {
    const source = sourceFromUnknown(candidate)
    if (source) {
      sources.push(source)
    }
  }

  const data = isRecord(unknownMessage.data) ? unknownMessage.data : null
  if (data) {
    for (const candidate of toArray(data.sources)) {
      const source = sourceFromUnknown(candidate)
      if (source) {
        sources.push(source)
      }
    }

    for (const candidate of toArray(data.citations)) {
      const source = sourceFromUnknown(candidate)
      if (source) {
        sources.push(source)
      }
    }
  }

  return dedupeSources(sources)
}
