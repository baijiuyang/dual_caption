import time
import argparse
from openai import OpenAI, AsyncOpenAI
import srt
import asyncio

client = OpenAI()
parser = argparse.ArgumentParser()

parser.add_argument("filename", type=str)
parser.add_argument("lang_a", type=str)
parser.add_argument("lang_b", type=str)

args = parser.parse_args()


def get_content_from_srt(raw_srt: str) -> list[str]:
    sub_generator = srt.parse(raw_srt)
    subs = list(sub_generator)
    lines = []
    for line in subs:
        lines.append(line.content)
    return lines


def add_second_subtitles(raw_srt: str, lines: list[str]) -> str:
    sub_generator = srt.parse(raw_srt)
    subs = list(sub_generator)
    for sub, line in zip(subs, lines):
        sub.content += f"\n{line}"
    return srt.compose(subs)


def load_srt(filename: str) -> str:
    with open(filename, "r", encoding="utf8") as f:
        raw_srt = f.read()
    return raw_srt


def save_srt(raw_srt: str, filename: str = "dual.srt") -> None:
    with open(filename, "w", encoding="utf8") as f:
        f.write(raw_srt)


def create_prompt(
    lang_a: str, lang_b: str, line: str, last_line: str, next_line: str
) -> str:
    return f"""Given the context "{last_line} {line} {next_line}", translate this {lang_a} subtitle to {lang_b}:\n{line}"""


def create_instruction() -> str:
    return "You are a subtitles translator."


def count_words(content: str) -> str:
    return content.count(" ")


def get_answer(instruction: str, prompt: str):
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        model="gpt-4o-mini",
        # model="gpt-4-turbo-2024-04-09",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        # max_tokens=4096,
        temperature=0.5,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content, response.usage.total_tokens


def main():
    raw_srt = load_srt(args.filename)
    lines = get_content_from_srt(raw_srt)
    instruction = create_instruction()
    total_results = []
    for i in range(len(lines)):
        if i > 450 and i % 450 == 1:
            print("Reached 450 RPM. Wait for 62s.")
            time.sleep(62)
        last_line = lines[i - 1] if i > 0 else ""
        line = lines[i]
        next_line = lines[i + 1] if i < len(lines) - 1 else ""
        prompt = create_prompt(args.lang_a, args.lang_b, line, last_line, next_line)
        total_results.append(get_answer(instruction, prompt))
    total_usage = 0
    lines = []
    for line, usage in total_results:
        lines.append(line)
        total_usage += usage

    print(f"total token used: {total_usage}")

    new_srt = add_second_subtitles(raw_srt, lines)
    save_srt(new_srt, args.filename[:-4] + "_output.srt")


if __name__ == "__main__":
    main()
    # raw_srt = load_srt("en.srt")
    # content = get_content_from_srt(raw_srt)
    # print(content)
