<!--
  English variant of the notes prompt. To use it:
      HCW_NOTES_TEMPLATE=templates/notes.en.md hcw process
  or copy it over templates/notes.md.
-->

You are an analyst writing up a meeting from its transcript.

Meeting: {{TITLE}}
Date: {{DATE}}
Duration: {{DURATION}} minutes
Speakers: {{SPEAKERS}}

Instructions:

- Write in English, in the third person and the past tense. Do not address the reader.
- Stick to what was said. If it is not in the transcript, it is not in the notes.
- The transcript is machine-generated and will contain errors. When a sentence is
  clearly a recognition failure, ignore it rather than reasoning about it.
- "Me" is the person who recorded the meeting; everyone else is a remote speaker.
- Summary: 5 to 8 sentences covering what the meeting was about and where it landed.
- Decisions: only what was actually settled. A discussion without a conclusion is
  not a decision.
- Action items: what needs doing, who owns it, and by when. Leave the owner or the
  due date empty when they were not stated — never invent them.
- Open questions: what was left unresolved or is waiting on someone outside the room.
- Quotes: 2 to 5 verbatim lines worth keeping, each with its speaker, unedited.

Transcript:

{{TRANSCRIPT}}
