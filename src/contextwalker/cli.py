from contextwalker.pipeline import build_system, search


def main():
    print("\n" + "=" * 80)
    print("CONTEXTUAL + HYBRID + AGENTIC RAG")
    print("=" * 80)

    index, vector_row_map, bm25, bm25_row_map, chunks = build_system()

    print("\n" + "=" * 80)
    print("RAG SYSTEM READY")
    print("=" * 80)
    print("\nType 'exit' to quit.")

    while True:
        query = input("\nAsk a question: ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            break

        try:
            answer, results = search(
                query=query,
                index=index,
                vector_row_map=vector_row_map,
                bm25=bm25,
                bm25_row_map=bm25_row_map,
                chunks=chunks,
            )

            print("\n" + "=" * 80)
            print("FINAL ANSWER")
            print("=" * 80)
            print("\n" + answer)

            from rich.console import Console
            from rich.markdown import Markdown
            from rich.panel import Panel

            console = Console()
            console.print()
            console.print(
                Panel.fit(
                    "[bold]FINAL ANSWER[/bold]",
                    border_style="cyan",
                )
            )
            console.print(Markdown(answer))
        except Exception as error:
            print("\n[ERROR]", repr(error))


if __name__ == "__main__":
    main()
