import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import { describe, expect, it } from "vitest";
import remarkGfm from "remark-gfm";
import { markdownComponents } from "./AnswerCard";
import { remarkCitationChips } from "./citations";

describe("citation rendering", () => {
  it("renders citation chips inside GFM tables without duplicating cell text", () => {
    const answer = [
      "| Part | Detail |",
      "| --- | --- |",
      "| Saga | [architecture-guide.pdf, page 4] [E1] |",
    ].join("\n");
    const html = renderToStaticMarkup(
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkCitationChips]} components={markdownComponents}>
        {answer}
      </ReactMarkdown>,
    );
    expect(html).toContain("<th>Part</th>");
    expect(html).toContain("<td>Saga</td>");
    expect(html).toContain('class="citation-chip" title="architecture-guide.pdf, page 4"');
    expect(html).not.toContain("<th>PartPart</th>");
    expect(html).not.toContain("[E1]");
  });
});
