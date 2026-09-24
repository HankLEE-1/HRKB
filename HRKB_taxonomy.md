# Harmful Rhetorical Strategy Taxonomy (23 Categories)

This taxonomy informs the harmful prototype count $K_h{=}23$ used during offline
HRKB construction. It merges 18 propaganda techniques from Da San Martino et al.
(EMNLP 2019), the hate-speech rhetoric categories of Racism No Way, and
meme-specific patterns from recent surveys, then deduplicates overlapping
categories.

## A. Emotional Manipulation

| # | Strategy | Definition | Meme Example |
|---|----------|------------|--------------|
| 1 | **Loaded Language** | Using words with strong positive/negative emotional connotations to influence judgment | "These animals are invading our country" |
| 2 | **Appeal to Fear/Prejudice** | Gaining support by creating fear or exploiting existing prejudices | "If we don't stop them, they'll replace us" |
| 3 | **Flag-Waving** | Using patriotic sentiments to justify discriminatory behavior | "Real Americans don't look like that" |
| 4 | **Glittering Generalities** | Using vague positive words (e.g., freedom, purity) to guide identification | "Protecting our culture and values" |
| 5 | **Slogans** | Using short, powerful slogans to provoke emotional reactions | "Go back to where you came from" |

## B. Logical Fallacies

| # | Strategy | Definition |
|---|----------|------------|
| 6 | **Causal Oversimplification** | Attributing complex problems to a single cause (usually a specific group) |
| 7 | **Black-and-White Fallacy** | Presenting only two extreme choices, excluding middle ground |
| 8 | **Strawman** | Distorting the opponent's position and then attacking the distorted version |
| 9 | **Red Herring** | Introducing irrelevant topics to divert attention |
| 10 | **Whataboutism** | Deflecting criticism by pointing to the accuser's problems instead |
| 11 | **False Equivalence** | Equating fundamentally different things for comparison |

## C. Dehumanization & Othering

| # | Strategy | Definition |
|---|----------|------------|
| 12 | **Dehumanization** | Comparing people to animals, pests, diseases, or objects |
| 13 | **Demonization** | Portraying target groups as inherently evil, dangerous, or immoral |
| 14 | **Negative Stereotyping** | Attributing the same negative traits to all members of a group |
| 15 | **Scapegoating** | Blaming a specific group for all social problems |
| 16 | **Othering** | Emphasizing irreconcilable "us vs. them" differences |

## D. Discursive Strategies

| # | Strategy | Definition |
|---|----------|------------|
| 17 | **Dog-whistling / Coded Language** | Language that appears harmless on the surface but conveys hate signals to specific groups |
| 18 | **Ironic/Sarcastic Hate** | Packaging hatred in humor/irony, creating plausible deniability ("just joking") |
| 19 | **False Nostalgia** | "The past was better" narratives implying certain groups don't belong in the present |
| 20 | **Victim Blaming** | Attributing victims' suffering to the victims themselves |

## E. Information Manipulation

| # | Strategy | Definition |
|---|----------|------------|
| 21 | **Smears / Toxic Misinformation** | Spreading false information to associate target groups with crime or immorality |
| 22 | **Exaggeration / Minimization** | Exaggerating threats or downplaying the severity of atrocities |
| 23 | **Historical Revisionism / Genocide Denial** | Denying or distorting historical atrocities |

---

## Merged/Excluded Categories

The following categories from Da San Martino et al.'s original 20 propaganda
techniques are either rarely seen in memes or can be subsumed under the above
categories:

| Original Category | Merged Into |
|---|---|
| Repetition | Text-level technique, not a rhetorical strategy; not applicable in memes |
| Doubt | Merged into Smears |
| Appeal to Authority | Extremely rare in memes |
| Bandwagon | Merged into Flag-Waving |
| Thought-terminating Cliché | Merged into Black-and-White Fallacy |
| Obfuscation | Merged into Dog-whistling |
| Reductio ad Hitlerum | Subtype of Demonization |
| Name Calling/Labeling | Merged into Loaded Language |
| Incitement to Violence | Merged into Demonization |
| Racial/Religious Vilification | Superordinate concept, not a specific strategy |
| Sexual Objectification | Subtype of Dehumanization |
| Conspiracy Theory | Merged into Scapegoating + Toxic Misinformation |
