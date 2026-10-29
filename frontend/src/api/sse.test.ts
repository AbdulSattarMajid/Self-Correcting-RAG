import { describe, expect, it } from "vitest";
import { parseSSEStream } from "./sse";

function streamFrom(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(chunks: string[]) {
  const messages = [];
  for await (const message of parseSSEStream(streamFrom(chunks))) messages.push(message);
  return messages;
}

describe("parseSSEStream", () => {
  it("reassembles events split across chunks", async () => {
    await expect(collect(["event: node\ndata: {\"no", "de\":\"retrieve\"}\n\n"])).resolves.toEqual([
      { event: "node", data: '{"node":"retrieve"}' },
    ]);
  });

  it("joins multiple data lines with a newline", async () => {
    await expect(collect(["event: final\ndata: first\ndata: second\n\n"])).resolves.toEqual([
      { event: "final", data: "first\nsecond" },
    ]);
  });

  it("decodes unicode split inside a multibyte character", async () => {
    const bytes = new TextEncoder().encode("event: final\ndata: café 🔎\n\n");
    const splitAt = bytes.indexOf(0xf0) + 2;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, splitAt));
        controller.enqueue(bytes.slice(splitAt));
        controller.close();
      },
    });
    const messages = [];
    for await (const message of parseSSEStream(stream)) messages.push(message);
    expect(messages).toEqual([{ event: "final", data: "café 🔎" }]);
  });
});
