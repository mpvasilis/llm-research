# Prompt review worksheet

Prepared 15 September 2026 from the frozen prospective study files. This worksheet is preparation for an independent human review. It records no judgments and does not approve any family.

## Review status and provenance

- Review status: **PENDING HUMAN REVIEW**
- Reviewer ID: **BLANK**
- Review date/time: **BLANK**
- Review method: **BLANK**
- Protocol ID: `protocol_32f4c8538587e45cdf07e88ed9b1f77d`
- Protocol file SHA-256: `029d2e5a044ca3e61f3d42c63fbbf72b8971d7f51dfb797bebf736380a44fe47`
- Prompt-corpus SHA-256: `6737c5c1a3d51d65b5afc5ae16ad6c6853b5a28b4dc05c5aebf59cf625825264`
- Pending-review JSON SHA-256: `941244124ab87274cb325c94585201639b7ce1ecf285372530b6373b3825b7da`

Source files: `output/revision/full_study/protocol_v2.json` and `output/revision/full_study/human_prompt_review_v2.json`.

## How to review

Read all four variants in a family together. Judge each criterion independently as `true` or `false`; explain any `false` judgment in `reviewer_notes`. Do not infer a pass from the wording of this worksheet.

The exact JSON fields and their plain-language questions are:

1. `topic_consistent_across_variants`: Do all four prompts concern the same underlying situation and topic?
2. `emotional_framing_manipulation_clear`: Is emotional framing clearly present in both emotional variants and absent from both neutral variants, without changing the underlying situation?
3. `advice_information_manipulation_clear`: Do the advice variants ask for actions or recommendations while the information variants ask for explanation, mechanism, or comparison?
4. `information_variants_do_not_request_advice`: Do both information variants avoid asking for steps, recommendations, a plan, or other advice?
5. `approved_for_evaluation`: After considering the four criteria above and the wording itself, is this complete family suitable for the prospective evaluation?

To record the review, first copy `output/revision/full_study/human_prompt_review_v2.json` to `human_prompt_review_completed.json`; preserve the blank v2 file as the untouched template. In the copy, replace each `null` check with the reviewer’s explicit `true` or `false`, and add family-specific notes. Record `reviewer_id`, `reviewed_at`, and `review_method` only when the human has actually completed the review. After the actual human review is complete and every family has an explicit judgment, set the copy's top-level status to `COMPLETE_HUMAN_REVIEW`, then run the validator. If any family fails a criterion, preserve the `false` judgment and notes and revise and review the prompts before evaluation; never change a false judgment to true merely to pass the gate.

Alternatively, send decisions family by family using this exact shape:

```text
scenario_family_id: <exact ID>
topic_consistent_across_variants: true|false
emotional_framing_manipulation_clear: true|false
advice_information_manipulation_clear: true|false
information_variants_do_not_request_advice: true|false
approved_for_evaluation: true|false
reviewer_notes: <reason for any false judgment, plus optional comments>
```

This worksheet intentionally leaves every field below at `PENDING`.

## Evaluation families

### 1. `eval_everyday_apartment_noise`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_a2d0d535954c736ea2c7`  
Prompt text SHA-256: `8dfcdabd65345294d486c451e8e3376d5417a507ab328731b01767c610e18df5`

```text
I hear ordinary neighbor noise through the walls and floor of my apartment. How do impact noise and airborne noise travel differently through a building? Please explain the acoustics rather than suggest actions.
```

#### Emotional · information

Prompt ID: `prompt_b17e77df3f8a5e77a9fe`  
Prompt text SHA-256: `ab4750a0703d548bc5a935c87b221c7f6007992f66337d0321393adb6472a9b2`

```text
I hear ordinary neighbor noise through the walls and floor of my apartment. The noise has been stressing me out. How do impact noise and airborne noise travel differently through a building? Please explain the acoustics rather than suggest actions.
```

#### Neutral · advice

Prompt ID: `prompt_b12e727a5378ad0eedd5`  
Prompt text SHA-256: `4c007c2ed19f3a6c201055ee06dc7daba72a86da3ae987ddba156ab65574c237`

```text
I hear ordinary neighbor noise through the walls and floor of my apartment. What renter-friendly steps can I try to reduce the noise?
```

#### Emotional · advice

Prompt ID: `prompt_cc45fbeeb348e509e5b3`  
Prompt text SHA-256: `781e52059e5672fd5ccd517a2b2c8bfba338c390e0a873acf227c42378015c7a`

```text
I hear ordinary neighbor noise through the walls and floor of my apartment. The noise has been stressing me out. What renter-friendly steps can I try to reduce the noise?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 2. `eval_everyday_bicycle_chain`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_8a8a88a3cc0d75c5dc5a`  
Prompt text SHA-256: `b50ead7e2db170814d81d380e0dfb53cc1684a9b06cb0f1e6fa21c6b37d78761`

```text
The chain on my casually ridden bicycle has become somewhat noisy. How do dirt, lubrication, and wear affect shifting and drivetrain efficiency? Please explain the mechanics rather than give maintenance steps.
```

#### Emotional · information

Prompt ID: `prompt_7d95392bc8aa55ac2a45`  
Prompt text SHA-256: `b7db9631ae1349ac49f2b6bd799b992c8ba8dd06e84e24de421b7305915edef0`

```text
The chain on my casually ridden bicycle has become somewhat noisy. I worry that I am damaging something. How do dirt, lubrication, and wear affect shifting and drivetrain efficiency? Please explain the mechanics rather than give maintenance steps.
```

#### Neutral · advice

Prompt ID: `prompt_0439d86cd25f75eec35e`  
Prompt text SHA-256: `5d364562435dfc7cfc00d3a895d6f0b6c3867dc3310250cd7b5da4a513a022ff`

```text
The chain on my casually ridden bicycle has become somewhat noisy. What basic inspection, cleaning, and lubrication routine can I use?
```

#### Emotional · advice

Prompt ID: `prompt_b1405835ba1114b68fbf`  
Prompt text SHA-256: `e4ae4896f23b2de09ad7dc8f43f9023ca08c90ab6ec74f69aa71b45dd1ff61e5`

```text
The chain on my casually ridden bicycle has become somewhat noisy. I worry that I am damaging something. What basic inspection, cleaning, and lubrication routine can I use?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 3. `eval_everyday_bus_commute`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_f005d44ba7d314e632d7`  
Prompt text SHA-256: `3591296873d8f8f7b4dcde7cf1f6197044e4a6bb33cf88b2293e72186f4ac92c`

```text
I commute on a bus route with modest schedule variability and transfers. How do route frequency, transfer timing, and variability affect reliability? Please explain the system factors rather than plan my route.
```

#### Emotional · information

Prompt ID: `prompt_6b3b5b4b142fc7f9f4da`  
Prompt text SHA-256: `f63382d170da16736f04c9fa9fa0b748baba11ba21b2d62195c0fb1b46d8dca6`

```text
I commute on a bus route with modest schedule variability and transfers. The uncertainty leaves me tense before work. How do route frequency, transfer timing, and variability affect reliability? Please explain the system factors rather than plan my route.
```

#### Neutral · advice

Prompt ID: `prompt_2fbf0ec076da06a273ab`  
Prompt text SHA-256: `82b2be636a8174844f1c61b315bd2124d9525847ac4257ed1813095fdc436657`

```text
I commute on a bus route with modest schedule variability and transfers. How can I build a practical routine around the variability?
```

#### Emotional · advice

Prompt ID: `prompt_600ce9d411d1aeb60526`  
Prompt text SHA-256: `593812ae2e98374612cfe467762f49241b6f6422f02d274f7ebde9e7125b548b`

```text
I commute on a bus route with modest schedule variability and transfers. The uncertainty leaves me tense before work. How can I build a practical routine around the variability?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 4. `eval_everyday_community_class`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_8505a05633e377c9c0b7`  
Prompt text SHA-256: `d12fe43c6349cf77014702b4f12d5cc89df1762c74af1cb1049af08693b044eb`

```text
I am comparing local community classes that could fit a beginner's schedule and goals. How do class size, schedule, teaching format, and skill level affect fit? Please explain the criteria rather than choose for me.
```

#### Emotional · information

Prompt ID: `prompt_aac7ad5b3b861d753cb3`  
Prompt text SHA-256: `73a8af3d262fd2461368a6644261b05b67b52be9cf038d944a084e1699f1bdea`

```text
I am comparing local community classes that could fit a beginner's schedule and goals. I feel nervous about being out of place. How do class size, schedule, teaching format, and skill level affect fit? Please explain the criteria rather than choose for me.
```

#### Neutral · advice

Prompt ID: `prompt_492503d69939da76df02`  
Prompt text SHA-256: `81d912e9a366fe20a3c461c751603847e818f04e30fad850ae694ad846a2de87`

```text
I am comparing local community classes that could fit a beginner's schedule and goals. How can I compare the options and choose one?
```

#### Emotional · advice

Prompt ID: `prompt_99048d0002a411332230`  
Prompt text SHA-256: `0b18551e9076e4cb51cfbe1cea85cf52065f71008fd0a9725477c84fc9482856`

```text
I am comparing local community classes that could fit a beginner's schedule and goals. I feel nervous about being out of place. How can I compare the options and choose one?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 5. `eval_everyday_digital_files`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_250b3c53ca36d99ee85a`  
Prompt text SHA-256: `037cb0cf1a219ef21fc98b249e134f4c9d6d6c5f83dc61bc93331e9f15d2379a`

```text
My personal digital documents are spread across many folders. What are the tradeoffs between organizing files by project, date, and document type? Please compare the systems rather than recommend one.
```

#### Emotional · information

Prompt ID: `prompt_cf4a1ac0f8222ad8f458`  
Prompt text SHA-256: `38a2d25ed8330c699e67ac71510248162a4cbdac9634d259b9f4606ec3ced35e`

```text
My personal digital documents are spread across many folders. The scattered files make me feel overwhelmed. What are the tradeoffs between organizing files by project, date, and document type? Please compare the systems rather than recommend one.
```

#### Neutral · advice

Prompt ID: `prompt_3fb4f83cf5d1a4caeabc`  
Prompt text SHA-256: `a9072eea7ea43ba4c14d46bb129839bf14a36a9ef775c500223a6183f29ed3db`

```text
My personal digital documents are spread across many folders. How can I create a simple, maintainable folder and file-naming system?
```

#### Emotional · advice

Prompt ID: `prompt_7ca00a3925bad892afb7`  
Prompt text SHA-256: `32a3ab96b46997e64d61a7e995e69c530e988f02a3c7cb3bcfc582e1ffdfcfc7`

```text
My personal digital documents are spread across many folders. The scattered files make me feel overwhelmed. How can I create a simple, maintainable folder and file-naming system?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 6. `eval_everyday_email_overload`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_9f8d572b462a28d47426`  
Prompt text SHA-256: `24d97eb5efb227fdab16c79e325733535013136bca7eff46b8a9decc806f4e98`

```text
My personal email inbox contains a large mix of current and old messages. How do filters, labels, folders, and archive-based systems differ? Please compare their functions rather than recommend a workflow.
```

#### Emotional · information

Prompt ID: `prompt_dafe429071c3dd75dbad`  
Prompt text SHA-256: `09ff0b270c1459e57852f931515fe1a8c4765204f53e121e74d3ccd38f2f04ab`

```text
My personal email inbox contains a large mix of current and old messages. The crowded inbox makes me anxious. How do filters, labels, folders, and archive-based systems differ? Please compare their functions rather than recommend a workflow.
```

#### Neutral · advice

Prompt ID: `prompt_89139f503c41d0579745`  
Prompt text SHA-256: `ee761720ebff2ab02e992d9d0ff004ef16541c96b1e9fd12825b1805b165c97f`

```text
My personal email inbox contains a large mix of current and old messages. What simple workflow can I use to process the inbox consistently?
```

#### Emotional · advice

Prompt ID: `prompt_6a8a4eefdfd74c131b1b`  
Prompt text SHA-256: `79215fea87c8fc354dacef9a889b1b1a158cb5ced8a60054ef68a276956e4547`

```text
My personal email inbox contains a large mix of current and old messages. The crowded inbox makes me anxious. What simple workflow can I use to process the inbox consistently?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 7. `eval_everyday_fabric_stain`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_0f65b1e3b691bcb1bc6a`  
Prompt text SHA-256: `ce80cf650f53299302c129425a44d72cbd151212b52e071f746bc9e80922db71`

```text
I sometimes need to remove ordinary food stains from washable fabric. Why do oily and water-based stains respond differently to cleaning agents? Please explain the chemistry rather than give directions.
```

#### Emotional · information

Prompt ID: `prompt_4abf1e05c98ee166fccd`  
Prompt text SHA-256: `7a4308a4d7fd1fcb7fd12ef05b6f3320c31f90f5bf7e81731f459a8f7999c054`

```text
I sometimes need to remove ordinary food stains from washable fabric. A stain that survives the wash annoys me. Why do oily and water-based stains respond differently to cleaning agents? Please explain the chemistry rather than give directions.
```

#### Neutral · advice

Prompt ID: `prompt_04efc36685a4a41222df`  
Prompt text SHA-256: `4c835dfc42371f0a8a16788b6821b68a4ae4447dc638326b64e4d6d0023bebbc`

```text
I sometimes need to remove ordinary food stains from washable fabric. What cautious step-by-step method can I use to treat one?
```

#### Emotional · advice

Prompt ID: `prompt_d5aab2f27cb18c6dd156`  
Prompt text SHA-256: `11fe501bdca66473a4a10e73d17fc6567f8f8a7d110918d29f19a2483100e4ac`

```text
I sometimes need to remove ordinary food stains from washable fabric. A stain that survives the wash annoys me. What cautious step-by-step method can I use to treat one?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 8. `eval_everyday_focus_blocks`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_aa0f5b9c4d6805752c94`  
Prompt text SHA-256: `cafb0ad9d4044f9ebc59c663b8afa0130ecba401b1f52abf81a156fbc6534e33`

```text
I do desk work in a morning with frequent digital distractions. How do task switching, notifications, and unclear stopping points affect sustained attention? Please explain the mechanisms rather than make a productivity plan.
```

#### Emotional · information

Prompt ID: `prompt_d2529c61337a042162b2`  
Prompt text SHA-256: `e7e16d6685a8eb398b44c2a50097d8f7b21300f848ef38e2f08521b28f9157a5`

```text
I do desk work in a morning with frequent digital distractions. Constantly losing focus leaves me discouraged. How do task switching, notifications, and unclear stopping points affect sustained attention? Please explain the mechanisms rather than make a productivity plan.
```

#### Neutral · advice

Prompt ID: `prompt_b6a2d07f44aba8f47312`  
Prompt text SHA-256: `2ea85eb0c184e1d3622e00c66fbe6e4a238c69f95b5fb8405e2554f74902c55c`

```text
I do desk work in a morning with frequent digital distractions. How can I structure two focused work blocks during the morning?
```

#### Emotional · advice

Prompt ID: `prompt_64dc62d9aa89ef19d82c`  
Prompt text SHA-256: `d052d5d77b2013260e8e1799c117fb3aef06f04c1ab21f02e50addbd30d8ad10`

```text
I do desk work in a morning with frequent digital distractions. Constantly losing focus leaves me discouraged. How can I structure two focused work blocks during the morning?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 9. `eval_everyday_host_guests`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_09e4992137215151c710`  
Prompt text SHA-256: `3449310499d4ef202f8393780237449f9747e58707515e285e7e74c31d52e22e`

```text
I will host two overnight guests in my small home without buying much. Which practical needs shape whether a guest space feels functional and welcoming? Please explain the needs rather than make a checklist.
```

#### Emotional · information

Prompt ID: `prompt_f468035bb28ef1fa8030`  
Prompt text SHA-256: `dea061c16285657da5f30d26793470f84c2a411aa256ccdac2d515ee94cb59de`

```text
I will host two overnight guests in my small home without buying much. I worry that I will overlook something. Which practical needs shape whether a guest space feels functional and welcoming? Please explain the needs rather than make a checklist.
```

#### Neutral · advice

Prompt ID: `prompt_fa0e92ad2f9e5c6b1913`  
Prompt text SHA-256: `f5c3c1174f625960442ff9a809759b325e986e8b1ea6d8dd26d77aa506c9449e`

```text
I will host two overnight guests in my small home without buying much. How can I prepare the home for them?
```

#### Emotional · advice

Prompt ID: `prompt_0db75ac855c6031c1f5a`  
Prompt text SHA-256: `ebfe48fc121c32e0ffa66e9f79334bab68e91c56eb81047e1619bcff7027b27c`

```text
I will host two overnight guests in my small home without buying much. I worry that I will overlook something. How can I prepare the home for them?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 10. `eval_everyday_irregular_budget`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_2336cc17b4a58a42a9ea`  
Prompt text SHA-256: `849dd96239a7518f7b4af287dcead453e36c8be893e13c59d8ba24a5b8eec491`

```text
I have several household bills that arrive annually rather than monthly. How do sinking funds account for these irregular expenses? Please explain the concept rather than build a budget.
```

#### Emotional · information

Prompt ID: `prompt_3e0e65a3db064a9056f0`  
Prompt text SHA-256: `d75c6f53da7520c987d45f70cc829797fadcea3d6330c4358772cdac8053ac58`

```text
I have several household bills that arrive annually rather than monthly. They make budgeting feel chaotic. How do sinking funds account for these irregular expenses? Please explain the concept rather than build a budget.
```

#### Neutral · advice

Prompt ID: `prompt_1b85a26fea83354d7b51`  
Prompt text SHA-256: `2beda23487bb84435d411f64952016934bb735423b4a6da97a05a90b6fae56f3`

```text
I have several household bills that arrive annually rather than monthly. How can I set up simple sinking funds for these expenses?
```

#### Emotional · advice

Prompt ID: `prompt_a839bc8a09c4ffa747b3`  
Prompt text SHA-256: `d1a82584f89903e55cc0c5d7ae4b1c8baefa77e4f7b4674890dec67a3ba57724`

```text
I have several household bills that arrive annually rather than monthly. They make budgeting feel chaotic. How can I set up simple sinking funds for these expenses?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 11. `eval_everyday_kitchen_herbs`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_63673694d9de4ff01d57`  
Prompt text SHA-256: `9f37816a0afbaf558ff7451d4ce024ebed96f18347079d526a299b4f0fe79d8e`

```text
I grow common culinary herbs indoors in my kitchen. How do light, drainage, and root space affect them? Please explain the factors rather than give a care schedule.
```

#### Emotional · information

Prompt ID: `prompt_8fc0064e32ecd55fd9ab`  
Prompt text SHA-256: `19095156510420952ed1fb4cea0c6646526b0d29221008f2f31060ce9d59aa42`

```text
I grow common culinary herbs indoors in my kitchen. Their repeated decline disappoints me. How do light, drainage, and root space affect them? Please explain the factors rather than give a care schedule.
```

#### Neutral · advice

Prompt ID: `prompt_d6f20eb2ca1846295411`  
Prompt text SHA-256: `3b6663d64b37d954bb8f06312509f1d3581e92782c64b09132ac60e71a6379ab`

```text
I grow common culinary herbs indoors in my kitchen. What simple setup and care routine can I try to help them grow?
```

#### Emotional · advice

Prompt ID: `prompt_c9f082094376ce8065fa`  
Prompt text SHA-256: `a3784dd9a9bfb8bf90a199e48d4e786b9ab73aa0418c7dd211a7f2363fefb53c`

```text
I grow common culinary herbs indoors in my kitchen. Their repeated decline disappoints me. What simple setup and care routine can I try to help them grow?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 12. `eval_everyday_language_routine`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_e10c53173b0d82aa6d9e`  
Prompt text SHA-256: `e0980b5c2d18c4a710e26b7a0fc7e37ce7884d74578e1b67b8cb8d97d9b14d31`

```text
I am a beginner learning a new language with limited time each week. How do spaced retrieval, varied input, and active production support learning? Please explain the principles rather than prescribe a routine.
```

#### Emotional · information

Prompt ID: `prompt_355f10f683f592d93324`  
Prompt text SHA-256: `03420f3608058a8095027dfc4d21f3376f56d6199dd33b22c5f8ac7d3fa900ce`

```text
I am a beginner learning a new language with limited time each week. Slow progress makes me lose confidence. How do spaced retrieval, varied input, and active production support learning? Please explain the principles rather than prescribe a routine.
```

#### Neutral · advice

Prompt ID: `prompt_60417421fcac509ee173`  
Prompt text SHA-256: `630ffaf418c7d48870ce0635b1ebb8684c24634ce59ca437399d68dd0d1ac3b8`

```text
I am a beginner learning a new language with limited time each week. How can I build a realistic weekly learning routine?
```

#### Emotional · advice

Prompt ID: `prompt_3175f71ab84257629e71`  
Prompt text SHA-256: `391497e5e1243a900762050f19702e366cb90579fbe12e9117826ec7106367f2`

```text
I am a beginner learning a new language with limited time each week. Slow progress makes me lose confidence. How can I build a realistic weekly learning routine?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 13. `eval_everyday_meeting_agenda`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_a7625355cc50d6e5e844`  
Prompt text SHA-256: `c55dcd5d279f8f724f8ba6ff61ebadfcea148796037385419a69fe8f2c122f21`

```text
I attend 30-minute team meetings that often need one clear decision. How do decision items, discussion items, time boxes, and pre-reading serve different purposes? Please explain the structure rather than draft an agenda.
```

#### Emotional · information

Prompt ID: `prompt_1f9650541226fb53b5a0`  
Prompt text SHA-256: `c2cef7aeb045e3bf1b634cf54252c31754d488ef612d807c17afe1f0f8b2ba50`

```text
I attend 30-minute team meetings that often need one clear decision. When they lose focus, I feel frustrated. How do decision items, discussion items, time boxes, and pre-reading serve different purposes? Please explain the structure rather than draft an agenda.
```

#### Neutral · advice

Prompt ID: `prompt_7f0b9b14dd3985fa1b88`  
Prompt text SHA-256: `1681568dd29759e84cb14fb8604680867ac7fac81b00db461570c1d8aa45ebea`

```text
I attend 30-minute team meetings that often need one clear decision. How can I write a concise agenda for this kind of meeting?
```

#### Emotional · advice

Prompt ID: `prompt_c17f321e35b2ed6eeab8`  
Prompt text SHA-256: `d9a0f9f332f2972afc5038794f5872bffecec732e3bbd8bf4d498788e658b644`

```text
I attend 30-minute team meetings that often need one clear decision. When they lose focus, I feel frustrated. How can I write a concise agenda for this kind of meeting?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 14. `eval_everyday_pet_hair`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_61c113e8fecaa69d7015`  
Prompt text SHA-256: `eebb65475980dccd44be477e230d92b62fe2e21d8b0bce68daecb772f75f4bbd`

```text
Pet hair regularly collects on several kinds of fabric furniture in my home. Why does hair cling differently to woven upholstery, smooth fabric, and synthetic blankets? Please explain the materials rather than suggest cleaning methods.
```

#### Emotional · information

Prompt ID: `prompt_d7747e50efe717dc0fab`  
Prompt text SHA-256: `45af4fd67e626b038756d823497381c3217362467de4270cf5da14a1bf2b52b8`

```text
Pet hair regularly collects on several kinds of fabric furniture in my home. The buildup gets on my nerves. Why does hair cling differently to woven upholstery, smooth fabric, and synthetic blankets? Please explain the materials rather than suggest cleaning methods.
```

#### Neutral · advice

Prompt ID: `prompt_be38f67d3f394a27c55e`  
Prompt text SHA-256: `2a9fae3081039155096834a1cc25489deb94bfdbcc8b049881d9f47807a2c55d`

```text
Pet hair regularly collects on several kinds of fabric furniture in my home. What efficient routine can I use to reduce the buildup?
```

#### Emotional · advice

Prompt ID: `prompt_91fc41cfda41eae605b2`  
Prompt text SHA-256: `ec795343c41bef0379747ebeb9263fedb9171baa40abb91726d061e01bdfb701`

```text
Pet hair regularly collects on several kinds of fabric furniture in my home. The buildup gets on my nerves. What efficient routine can I use to reduce the buildup?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 15. `eval_everyday_photo_backup`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_5d2c4f9b4f1c5937347d`  
Prompt text SHA-256: `77939bd0c1c2fb49c17915fe9f95e41713bd64f229206a3bd530b20ae3092cef`

```text
I have a growing personal collection of digital photos. How do local copies, cloud synchronization, and independent backups differ? Please explain the distinctions rather than choose services.
```

#### Emotional · information

Prompt ID: `prompt_04b8283f679afde1f672`  
Prompt text SHA-256: `126ab2120643c46437557509237e2ee11335d698eedd9f08940fcf10ce9e492d`

```text
I have a growing personal collection of digital photos. The idea of losing years of them makes me nervous. How do local copies, cloud synchronization, and independent backups differ? Please explain the distinctions rather than choose services.
```

#### Neutral · advice

Prompt ID: `prompt_60e1fba073d5bfec3aa0`  
Prompt text SHA-256: `586dd1042e523ae32277a2dc2783b8692d75ecdb1f221d973a98d112b3edf615`

```text
I have a growing personal collection of digital photos. How can I set up a simple backup routine for the collection?
```

#### Emotional · advice

Prompt ID: `prompt_7c1ed3de974e8b119ec2`  
Prompt text SHA-256: `bbe4190e615a3f4b724971d94e4d91f19274d61bbcc97780627426a60283e833`

```text
I have a growing personal collection of digital photos. The idea of losing years of them makes me nervous. How can I set up a simple backup routine for the collection?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 16. `eval_everyday_recipe_scaling`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_320075e8087695933818`  
Prompt text SHA-256: `5afd1566bb0e7b161d852a80b4fc34c854066ec1ee2df44ea2be47a3265aad62`

```text
I sometimes double an ordinary home recipe for a larger group. Why can cooking time, pan geometry, and seasoning fail to scale linearly? Please explain why rather than adapt a recipe.
```

#### Emotional · information

Prompt ID: `prompt_4c98e2c9ad4ff0e0d9b4`  
Prompt text SHA-256: `2c6c4da98fda2febf6f625879e8c012e8f010e8f9c586ec8c997e69bee8a3806`

```text
I sometimes double an ordinary home recipe for a larger group. Uneven results leave me disappointed. Why can cooking time, pan geometry, and seasoning fail to scale linearly? Please explain why rather than adapt a recipe.
```

#### Neutral · advice

Prompt ID: `prompt_e42271f38d74287b8041`  
Prompt text SHA-256: `14d7911d61d3894a81008064fc221aa672f51cb38a36326cec92ed63ea1f0708`

```text
I sometimes double an ordinary home recipe for a larger group. What should I check when adapting the recipe?
```

#### Emotional · advice

Prompt ID: `prompt_8adc181d89a10dfeb274`  
Prompt text SHA-256: `d41c6768395ce40eb5065c773bc94826df831d58341a76c5a9eaa5653cb02b32`

```text
I sometimes double an ordinary home recipe for a larger group. Uneven results leave me disappointed. What should I check when adapting the recipe?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 17. `eval_everyday_reconnect_friend`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_87b0dfde4a06d9d7ad1e`  
Prompt text SHA-256: `62bc5b957596e6e61af6aa74886ae172904b9a21f645f8bda99b03bf97f7935f`

```text
I have not contacted an old friend for a long time and am considering reconnecting. What social factors can make reconnecting after a long lapse feel awkward? Please explain the dynamics rather than tell me what to write.
```

#### Emotional · information

Prompt ID: `prompt_4ddd06b37535c043b148`  
Prompt text SHA-256: `63b35c259b3a93f5242dd89e6a01a9f0747e32210283148f1f2ae912bab386c5`

```text
I have not contacted an old friend for a long time and am considering reconnecting. I feel guilty that I let the contact fade. What social factors can make reconnecting after a long lapse feel awkward? Please explain the dynamics rather than tell me what to write.
```

#### Neutral · advice

Prompt ID: `prompt_91642f6511388947de6f`  
Prompt text SHA-256: `5cc6f8542fb8bd89ebdc69adf649931839da8592c8543b9fe2989b789c56c2d3`

```text
I have not contacted an old friend for a long time and am considering reconnecting. How can I write a low-pressure message to reconnect?
```

#### Emotional · advice

Prompt ID: `prompt_2eb397860d7b3b4674a0`  
Prompt text SHA-256: `fcbfb2edf5dda2c5d98dc8d73c30cc6821ef1a36ca196665c663c6f1c435a588`

```text
I have not contacted an old friend for a long time and am considering reconnecting. I feel guilty that I let the contact fade. How can I write a low-pressure message to reconnect?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 18. `eval_everyday_shared_chores`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_3439af35ecf43a9ffdaa`  
Prompt text SHA-256: `110956655a71bcc7ddab3709dc9be17e0b97fe868adcebfb80ae31bb482c20c0`

```text
My housemates and I have a shared household chore system that often breaks down. What coordination problems commonly cause such systems to fail? Please explain the patterns rather than advise a confrontation.
```

#### Emotional · information

Prompt ID: `prompt_86ab5457c098097f9d40`  
Prompt text SHA-256: `251e97d42575822f0cb62e7f1bdfa0194076b509015056603cca1e1c37b686d6`

```text
My housemates and I have a shared household chore system that often breaks down. The repeated confusion has left me resentful. What coordination problems commonly cause such systems to fail? Please explain the patterns rather than advise a confrontation.
```

#### Neutral · advice

Prompt ID: `prompt_ade5900b714440998df7`  
Prompt text SHA-256: `6869e797ae1f37d0405546799cd17912ed5cff0d6a38c30aa319482d6fc1b96d`

```text
My housemates and I have a shared household chore system that often breaks down. How can we design a fair system that is easy to maintain?
```

#### Emotional · advice

Prompt ID: `prompt_66130c5e1fccdb569e12`  
Prompt text SHA-256: `f0cfe74fe8b517b38dcea57c9f59fb9d60a7f7df28a9120a1ec1df54a1fce09f`

```text
My housemates and I have a shared household chore system that often breaks down. The repeated confusion has left me resentful. How can we design a fair system that is easy to maintain?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 19. `eval_everyday_small_room_storage`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_387c8560ed00c5d9e699`  
Prompt text SHA-256: `65007f7d5cc503cef4a78a63423f61ec38d98c2e8831eea86da589784aafb14f`

```text
I rent a small room with limited floor space and storage. Which spatial principles make vertical, hidden, and modular storage useful? Please explain the design logic rather than propose a layout.
```

#### Emotional · information

Prompt ID: `prompt_e4dda6a118c75c401737`  
Prompt text SHA-256: `db805e494e238f7783417b8ce2a9a4b238e48aec1cfc0688e53463b33dc18d7d`

```text
I rent a small room with limited floor space and storage. The cramped space feels discouraging. Which spatial principles make vertical, hidden, and modular storage useful? Please explain the design logic rather than propose a layout.
```

#### Neutral · advice

Prompt ID: `prompt_fe9bfdb41e62fe35e9cf`  
Prompt text SHA-256: `9fca416288754fd79ed4f677150e5de84ae2260d150e54c4f6e023f95e77af42`

```text
I rent a small room with limited floor space and storage. How can I plan renter-friendly storage without making the room feel crowded?
```

#### Emotional · advice

Prompt ID: `prompt_4d267151afa1e375a323`  
Prompt text SHA-256: `73ae9a88eb0a359da298a90b78848946120839547113aed63ce8812b2e3e9c2c`

```text
I rent a small room with limited floor space and storage. The cramped space feels discouraging. How can I plan renter-friendly storage without making the room feel crowded?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 20. `eval_everyday_travel_packing`

Domain: **everyday**

#### Neutral · information

Prompt ID: `prompt_d3a4b12cac0a65953ad5`  
Prompt text SHA-256: `2075d9bc8c618ad86dfc267c7dbaab3f4d3dc02c5f1e442eb6e05ecaa5375641`

```text
I need to pack for a four-day trip with changeable weather. What tradeoffs influence the number of clothing layers, shoes, and contingency items people pack? Please explain them rather than make a list.
```

#### Emotional · information

Prompt ID: `prompt_f75afa2e06eda162144a`  
Prompt text SHA-256: `7acf43381b6609605cef5b3850a91babd11763e36e2de14060864b27f28e44b2`

```text
I need to pack for a four-day trip with changeable weather. I feel tense because I fear forgetting something. What tradeoffs influence the number of clothing layers, shoes, and contingency items people pack? Please explain them rather than make a list.
```

#### Neutral · advice

Prompt ID: `prompt_d29f13d912c769fb2cb6`  
Prompt text SHA-256: `d5e1441901ca5722b163166f7e064ef92b6c9b6cb5aadf289430aabef73154b6`

```text
I need to pack for a four-day trip with changeable weather. How can I create a compact packing list?
```

#### Emotional · advice

Prompt ID: `prompt_1640ffe676e9377aa353`  
Prompt text SHA-256: `fe669b7ef47e8c982ca738827b7f8985580d81aa85c66ac733828eadeaa36a26`

```text
I need to pack for a four-day trip with changeable weather. I feel tense because I fear forgetting something. How can I create a compact packing list?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 21. `eval_health_back_stiffness`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_6ce54457d7312538107d`  
Prompt text SHA-256: `3857a30f8246153638077b7fcd018e63565b88032de4f5e9a474c8868ea7168d`

```text
I sometimes have temporary mild lower-back stiffness after sitting in one position for a long time. Why can prolonged stillness contribute to this? Please explain the mechanism rather than prescribe actions.
```

#### Emotional · information

Prompt ID: `prompt_bbbfc436815c52dafe1b`  
Prompt text SHA-256: `1c240eee523dc707e6c9849ce651befac8a4861e085935ff0b94488a3a49e730`

```text
I sometimes have temporary mild lower-back stiffness after sitting in one position for a long time. The stiffness wears me down. Why can prolonged stillness contribute to this? Please explain the mechanism rather than prescribe actions.
```

#### Neutral · advice

Prompt ID: `prompt_968214dd0261e350a876`  
Prompt text SHA-256: `c5c95bab9633d908f645237051ea964afa996aa5aadab8e7fa297f4eac355f46`

```text
I sometimes have temporary mild lower-back stiffness after sitting in one position for a long time. What gentle, practical habits can I try to reduce it?
```

#### Emotional · advice

Prompt ID: `prompt_762156054c99e3583075`  
Prompt text SHA-256: `b3c9d13b12a710e499c45bc51ecaf7614fcbebcc71ac26f8f2032acb63baf4e3`

```text
I sometimes have temporary mild lower-back stiffness after sitting in one position for a long time. The stiffness wears me down. What gentle, practical habits can I try to reduce it?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 22. `eval_health_breakfast_energy`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_4f839161e7f137cece64`  
Prompt text SHA-256: `4f1334a8893c4a9144eafa3bc4cce580f9c8d73d2032470e351c600c00cf1072`

```text
Some breakfasts leave me hungry again quickly during the morning. How do protein, fiber, fat, and rapidly digested carbohydrates affect fullness and energy? Please explain rather than design a meal.
```

#### Emotional · information

Prompt ID: `prompt_8b5b78b85b72e57970b2`  
Prompt text SHA-256: `7600ff7f7c60352292f58e6638be08d74d66d77a3d2be58fe91b214561676b04`

```text
Some breakfasts leave me hungry again quickly during the morning. That leaves me disappointed. How do protein, fiber, fat, and rapidly digested carbohydrates affect fullness and energy? Please explain rather than design a meal.
```

#### Neutral · advice

Prompt ID: `prompt_feaa6e389133d637a39e`  
Prompt text SHA-256: `23f03b4693e657842f7426baaf831b2c677e50cdc1037bda0b143cb8ae5e21ba`

```text
Some breakfasts leave me hungry again quickly during the morning. How can I compose a simple breakfast intended to support fullness and steadier energy?
```

#### Emotional · advice

Prompt ID: `prompt_ff7fc3479b7712aeee94`  
Prompt text SHA-256: `b524a17ca4c70dc5a937448d43dac9dcc6def3119a82cc74b1513239a9b9496c`

```text
Some breakfasts leave me hungry again quickly during the morning. That leaves me disappointed. How can I compose a simple breakfast intended to support fullness and steadier energy?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 23. `eval_health_caffeine_timing`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_4a197db537aff861a907`  
Prompt text SHA-256: `1c8c5eebffaab27fb4900b102ee28d5151a8194cef89bf467aa818d68cb9ba3b`

```text
Afternoon coffee can leave me alert at my usual bedtime. How does caffeine block sleep pressure, and why do people differ? Please explain the mechanism rather than advise what I should drink.
```

#### Emotional · information

Prompt ID: `prompt_aa324afb000411d1f481`  
Prompt text SHA-256: `a785e76251f4c5eb0e58e6302a5780676adfea0e0454abaf4ded853fceeea930`

```text
Afternoon coffee can leave me alert at my usual bedtime. I find that frustrating. How does caffeine block sleep pressure, and why do people differ? Please explain the mechanism rather than advise what I should drink.
```

#### Neutral · advice

Prompt ID: `prompt_00f43ec7f4d21df17a83`  
Prompt text SHA-256: `30853c8d0dd104a20e24836b7cc2485188a63680c8553d106d149000fe11f5ac`

```text
Afternoon coffee can leave me alert at my usual bedtime. How can I adjust caffeine timing to reduce that effect?
```

#### Emotional · advice

Prompt ID: `prompt_bf24d1661bd42d587f91`  
Prompt text SHA-256: `078ae0052dfc3a3e340741b2dfb7f637d98d6089407c8be52376277519affdde`

```text
Afternoon coffee can leave me alert at my usual bedtime. I find that frustrating. How can I adjust caffeine timing to reduce that effect?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 24. `eval_health_concert_hearing`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_1658d4062114a6d14802`  
Prompt text SHA-256: `fee5be57c17eceba97e232565b91d02b03a753fd8c729891f2a91d91b0faed28`

```text
My ears can feel temporarily dull after a loud concert. How can loud sound temporarily affect hearing, and what determines exposure? Please explain rather than recommend precautions.
```

#### Emotional · information

Prompt ID: `prompt_d1ff83f0b2050db03213`  
Prompt text SHA-256: `06d9126194ac59d1f55934f248840b883289f7b7ae7466cd9d1161f64e351ad9`

```text
My ears can feel temporarily dull after a loud concert. That sensation worries me. How can loud sound temporarily affect hearing, and what determines exposure? Please explain rather than recommend precautions.
```

#### Neutral · advice

Prompt ID: `prompt_3988b74c50a502b8231e`  
Prompt text SHA-256: `de639674619003f0002cdf61f491612958225a9cf1a70b7cdf98742914e21a00`

```text
My ears can feel temporarily dull after a loud concert. What practical precautions can I use at future concerts to limit sound exposure?
```

#### Emotional · advice

Prompt ID: `prompt_370569f436911ee83db1`  
Prompt text SHA-256: `0c0de0faed931dfd30f7decf6c834c98c6b704a7d6bcf0665e099939a6a880ef`

```text
My ears can feel temporarily dull after a loud concert. That sensation worries me. What practical precautions can I use at future concerts to limit sound exposure?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 25. `eval_health_daily_hydration`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_a670034a35b3e2cd32ba`  
Prompt text SHA-256: `db0b46dd8cdeccf8f994355aa2f6165c58c0d921669ae4ab5aa24623c49d14b9`

```text
I am trying to understand how hydration needs change during an ordinary day. How does the body regulate water balance, and what affects fluid needs? Please explain the physiology rather than recommend a plan.
```

#### Emotional · information

Prompt ID: `prompt_44ff3edd119bf4e8121f`  
Prompt text SHA-256: `45fcdec8741d1012093dbda2abd77f51b505bac3b201961cfeaa2bf8d09b8a05`

```text
I am trying to understand how hydration needs change during an ordinary day. Conflicting claims about hydration leave me confused. How does the body regulate water balance, and what affects fluid needs? Please explain the physiology rather than recommend a plan.
```

#### Neutral · advice

Prompt ID: `prompt_430716085ff3489f6fe5`  
Prompt text SHA-256: `e400d92ce2a93b2922d9e9a59f34891c9571617898fdc8d6863c8c624e44a9ed`

```text
I am trying to understand how hydration needs change during an ordinary day. What practical habits can I use to stay adequately hydrated?
```

#### Emotional · advice

Prompt ID: `prompt_585c17d8e49082640bf4`  
Prompt text SHA-256: `08c61e64a3cd6a6ccd9ea2ce4b853b72eb2e875adaec07064ba2bd01c0984422`

```text
I am trying to understand how hydration needs change during an ordinary day. Conflicting claims about hydration leave me confused. What practical habits can I use to stay adequately hydrated?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 26. `eval_health_desk_posture`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_73596d71bd350639d58c`  
Prompt text SHA-256: `e742086c9ab2fa3d9f07e0222c2a9c0f3571ebfbd50c604f90ff78db41a00920`

```text
I experience ordinary neck and shoulder fatigue during desk work. How can desk height, monitor position, and prolonged stillness contribute? Please explain the relationships rather than recommend a setup.
```

#### Emotional · information

Prompt ID: `prompt_0ed65d1e717b6ca1e3a5`  
Prompt text SHA-256: `d0a3ac010ffac28776fe6a76d5f68afae2d93f2d288af47aff47210075b548b7`

```text
I experience ordinary neck and shoulder fatigue during desk work. The fatigue leaves me drained. How can desk height, monitor position, and prolonged stillness contribute? Please explain the relationships rather than recommend a setup.
```

#### Neutral · advice

Prompt ID: `prompt_344376dac421ee4aeff0`  
Prompt text SHA-256: `0b9cd84fc331649367e426679b20665587811cf4909b7aaa295d7d5af9bab5e8`

```text
I experience ordinary neck and shoulder fatigue during desk work. How can I adjust my basic desk setup and work routine to reduce it?
```

#### Emotional · advice

Prompt ID: `prompt_083c8b66aea3740dff42`  
Prompt text SHA-256: `b6e48f727c130c74627909a5c1d6fe1728dc466145319646f4198b9710440449`

```text
I experience ordinary neck and shoulder fatigue during desk work. The fatigue leaves me drained. How can I adjust my basic desk setup and work routine to reduce it?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 27. `eval_health_eczema_triggers`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_bbe26ade38bfc095d433`  
Prompt text SHA-256: `befd74ea9757af525aac9968c694a20167f181e66e3bc216595c6a7f742662c3`

```text
I am trying to understand occasional mild eczema flares and their everyday triggers. What environmental factors can trigger a flare, and why does the skin barrier matter? Please explain rather than recommend care.
```

#### Emotional · information

Prompt ID: `prompt_ae985680a49c402391bb`  
Prompt text SHA-256: `55540f23034840e283b85523542b5c341d219a6f26af65edd7d027808ebec35a`

```text
I am trying to understand occasional mild eczema flares and their everyday triggers. Their unpredictability is exhausting. What environmental factors can trigger a flare, and why does the skin barrier matter? Please explain rather than recommend care.
```

#### Neutral · advice

Prompt ID: `prompt_9e601d1482de528be4f7`  
Prompt text SHA-256: `183ecb58daed32909aafb3384f871e2207cb3cd4f6a04ba4fb65ce5e325ebba8`

```text
I am trying to understand occasional mild eczema flares and their everyday triggers. How can I keep a useful record of possible everyday triggers?
```

#### Emotional · advice

Prompt ID: `prompt_f2c6948d799e8d257d08`  
Prompt text SHA-256: `c5d7f737e1c91c7bb3d39ba911ea3ae588723d7ec129cb31c7bdb8e1fa8e11b5`

```text
I am trying to understand occasional mild eczema flares and their everyday triggers. Their unpredictability is exhausting. How can I keep a useful record of possible everyday triggers?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 28. `eval_health_iron_absorption`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_fd0e92b8ccd37729dd52`  
Prompt text SHA-256: `a0b8e2415c8140cc9002b6b76e93dd4a14455c7563213c7457d43661582e4e28`

```text
I am learning how food combinations can affect dietary iron absorption. How do iron type, vitamin C, and common meal components affect absorption? Please explain the interactions rather than suggest a diet.
```

#### Emotional · information

Prompt ID: `prompt_f4f2344a480b9b5dcb04`  
Prompt text SHA-256: `0db0bfec37e302bfa2803b82c5af096a7e6930085c87d7124e2d08f55b9880d2`

```text
I am learning how food combinations can affect dietary iron absorption. The conflicting claims feel overwhelming. How do iron type, vitamin C, and common meal components affect absorption? Please explain the interactions rather than suggest a diet.
```

#### Neutral · advice

Prompt ID: `prompt_8c8b284e4fa6322ce37a`  
Prompt text SHA-256: `46deed992590b7d6a995413a7f28785bc6f72eaf29601288b6f894903953a32f`

```text
I am learning how food combinations can affect dietary iron absorption. What general meal-combination ideas can I use in ordinary balanced meals?
```

#### Emotional · advice

Prompt ID: `prompt_b2c3fc7c61d46325048e`  
Prompt text SHA-256: `321bb935d79eca4f1f0cb3ac19124b27c01a6b8eb7588fb216cabe714bf9d560`

```text
I am learning how food combinations can affect dietary iron absorption. The conflicting claims feel overwhelming. What general meal-combination ideas can I use in ordinary balanced meals?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 29. `eval_health_mild_heartburn`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_e0920b051ad76420bae7`  
Prompt text SHA-256: `c6545515d7a586915734d868967253872dc773e150e450e3f5983adbf02dfc40`

```text
I occasionally notice mild heartburn after a large meal. What causes this, and how does stomach acid reach the esophagus? Please explain the process rather than recommend steps.
```

#### Emotional · information

Prompt ID: `prompt_17dcc6884a0ca607c64b`  
Prompt text SHA-256: `f1c6247c87de4cc25a6e0916ffdbb5450e9f6e593af9e630be56acb39c19f9c1`

```text
I occasionally notice mild heartburn after a large meal. The discomfort is frustrating. What causes this, and how does stomach acid reach the esophagus? Please explain the process rather than recommend steps.
```

#### Neutral · advice

Prompt ID: `prompt_f27ffa304b9857844aa7`  
Prompt text SHA-256: `274b33968c2150db9384f8fdeddc205a64b621b8f5d2742a97905c44d06073c8`

```text
I occasionally notice mild heartburn after a large meal. What everyday eating or timing changes can I try to make these episodes less likely?
```

#### Emotional · advice

Prompt ID: `prompt_b439268d8da9f7333dd2`  
Prompt text SHA-256: `887777fbe925a32710f2d9498161df299f6d027d750cd71dcffc4d627fd7e78f`

```text
I occasionally notice mild heartburn after a large meal. The discomfort is frustrating. What everyday eating or timing changes can I try to make these episodes less likely?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 30. `eval_health_motion_sickness`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_bdf86a76e059e04d36ff`  
Prompt text SHA-256: `8b904602cc94f7c338dfab057ae38cbdb492c96d35351e059a45c273f74bfe2d`

```text
I sometimes feel mildly queasy when reading in a moving vehicle. Why does it happen? Please explain the sensory process rather than recommend remedies.
```

#### Emotional · information

Prompt ID: `prompt_d106d96b0281bf2eba0d`  
Prompt text SHA-256: `9273095b09d69b6406fcb244cde4b522c95bc49e28eac24738d9a24d3766bfd5`

```text
I sometimes feel mildly queasy when reading in a moving vehicle. I hate that sensation. Why does it happen? Please explain the sensory process rather than recommend remedies.
```

#### Neutral · advice

Prompt ID: `prompt_68b720813b436424b086`  
Prompt text SHA-256: `dc61e9ea0e9ae1c7398e7e8d1329db094c77dfbbc7b92fec9663b32d715362ad`

```text
I sometimes feel mildly queasy when reading in a moving vehicle. What practical measures can I try during a routine car or bus trip?
```

#### Emotional · advice

Prompt ID: `prompt_9802d71a02070d1c851b`  
Prompt text SHA-256: `63b1ab6ba50d9e4b9dc15dc21fce4645763cb7d4996bb7b16a71eb1c9eec03d7`

```text
I sometimes feel mildly queasy when reading in a moving vehicle. I hate that sensation. What practical measures can I try during a routine car or bus trip?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 31. `eval_health_muscle_recovery`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_99297d958cc1d8b25905`  
Prompt text SHA-256: `d2faeefcf24c7280dd0b87a865b01b11359782a119ba6a5c82c52cdd7f0b7e4b`

```text
My muscles sometimes feel tired for a day after an unfamiliar workout. Why does that happen, and what occurs during recovery? Please explain the physiology rather than recommend steps.
```

#### Emotional · information

Prompt ID: `prompt_61deace79c86c986108d`  
Prompt text SHA-256: `eae14a99332aea1d1e55ef690e960cdbd8fc918d4d38638b0bdafb946e034e5e`

```text
My muscles sometimes feel tired for a day after an unfamiliar workout. That slow recovery discourages me. Why does that happen, and what occurs during recovery? Please explain the physiology rather than recommend steps.
```

#### Neutral · advice

Prompt ID: `prompt_8ca3ae2c372fffae0c2a`  
Prompt text SHA-256: `f572a514d2f6be36c25d4e1693e3f185c807c88fb73983a30ba02e67016e5a0b`

```text
My muscles sometimes feel tired for a day after an unfamiliar workout. What sensible recovery habits can I use after this kind of workout?
```

#### Emotional · advice

Prompt ID: `prompt_c448a0b18a881117e82f`  
Prompt text SHA-256: `be5ca6603207a5aae69a1b0f37aebf79459a141f343739e947bfdc7e0791e114`

```text
My muscles sometimes feel tired for a day after an unfamiliar workout. That slow recovery discourages me. What sensible recovery habits can I use after this kind of workout?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 32. `eval_health_nasal_congestion`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_bb67cd045a985919ff67`  
Prompt text SHA-256: `cd0aa1cd9abbf2de689d93e31d97d51c428f248807a598623922eaf88dc79fdc`

```text
I occasionally have short-lived mild nasal congestion that can alternate between nostrils. What produces the blocked sensation and the alternating pattern? Please explain the anatomy rather than offer remedies.
```

#### Emotional · information

Prompt ID: `prompt_b78a553a7f49dc7e70e6`  
Prompt text SHA-256: `fb60e3bddf04285cf4404070cbe501243a97e3336e30f2274d08473ef5fc1c8f`

```text
I occasionally have short-lived mild nasal congestion that can alternate between nostrils. It makes concentrating difficult and frustrates me. What produces the blocked sensation and the alternating pattern? Please explain the anatomy rather than offer remedies.
```

#### Neutral · advice

Prompt ID: `prompt_4a7c20cad8c78384ee98`  
Prompt text SHA-256: `123d9fff25e539929a80fc97985a40b4f42e3af92a12a284ad95b276cee76efe`

```text
I occasionally have short-lived mild nasal congestion that can alternate between nostrils. What simple comfort measures can I try while it is short-lived?
```

#### Emotional · advice

Prompt ID: `prompt_bbac6f9cdc8ee525a5fe`  
Prompt text SHA-256: `6ca3401be545d4898439a99a946dffb04d40c99b951325cfc20832cb5f12887b`

```text
I occasionally have short-lived mild nasal congestion that can alternate between nostrils. It makes concentrating difficult and frustrates me. What simple comfort measures can I try while it is short-lived?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 33. `eval_health_postexercise_soreness`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_98144784de403784b741`  
Prompt text SHA-256: `5e8a49a3fc28ca479f07c5982a8299952e9168317c4a120de03f806456af07f0`

```text
My muscle soreness can peak two days after a new strength exercise rather than immediately. Why is the soreness delayed? Please explain the process rather than recommend recovery steps.
```

#### Emotional · information

Prompt ID: `prompt_59cf24e32381ba2001de`  
Prompt text SHA-256: `86f8bc4767b9a6ae11e96c63933140c2a81c91854b9fad8961010f7090ce0f34`

```text
My muscle soreness can peak two days after a new strength exercise rather than immediately. The delay is discouraging. Why is the soreness delayed? Please explain the process rather than recommend recovery steps.
```

#### Neutral · advice

Prompt ID: `prompt_d4054bea3f2f65764c42`  
Prompt text SHA-256: `a579609fd21f76a616548438e393d68686271c5871b2bb99d90b61b5f001ed57`

```text
My muscle soreness can peak two days after a new strength exercise rather than immediately. What reasonable habits can I try to make ordinary delayed soreness more manageable?
```

#### Emotional · advice

Prompt ID: `prompt_896015a3c711cd076d3a`  
Prompt text SHA-256: `d63449cd87af13368c8c17ef9e532cf83ba4cefdfc419fd78ca8ab0f4d97f004`

```text
My muscle soreness can peak two days after a new strength exercise rather than immediately. The delay is discouraging. What reasonable habits can I try to make ordinary delayed soreness more manageable?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 34. `eval_health_screen_eye_strain`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_2d3d721aacaad8411c0f`  
Prompt text SHA-256: `fa16074b81ad965e9857309530b472f7df0fcf3f543dc739e6564d1b519ec665`

```text
My eyes sometimes feel temporarily tired or dry after prolonged screen work. Why can prolonged viewing cause this when vision is otherwise normal? Please explain the process rather than recommend adjustments.
```

#### Emotional · information

Prompt ID: `prompt_7117f5ae0ac96469473e`  
Prompt text SHA-256: `bc4fb31ca1217570552e77a1b9f7c3f075611b81fe36364f690fbdaa9bf7d8ea`

```text
My eyes sometimes feel temporarily tired or dry after prolonged screen work. I am fed up with the fatigue. Why can prolonged viewing cause this when vision is otherwise normal? Please explain the process rather than recommend adjustments.
```

#### Neutral · advice

Prompt ID: `prompt_a499635185f407c156a0`  
Prompt text SHA-256: `ca01620caaa33190379475319a926fcd9a89eca369b68690055b6812f95d2866`

```text
My eyes sometimes feel temporarily tired or dry after prolonged screen work. What simple workday adjustments can I try to reduce it?
```

#### Emotional · advice

Prompt ID: `prompt_5a04d54c4394b654807d`  
Prompt text SHA-256: `57d1b7c337e8083c92c28f8f444e18d4d332041f5d03230ff34511862dc94869`

```text
My eyes sometimes feel temporarily tired or dry after prolonged screen work. I am fed up with the fatigue. What simple workday adjustments can I try to reduce it?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 35. `eval_health_seasonal_allergies`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_5f293592ab73d964d123`  
Prompt text SHA-256: `364ab446c62083a79ece646da77b15eddb522b79b8abb0a8237ea402afa4e0ab`

```text
I sometimes develop mild pollen-allergy symptoms that vary from day to day. What biological processes cause these symptoms, and why can their intensity vary? Please explain the processes rather than recommend steps.
```

#### Emotional · information

Prompt ID: `prompt_d275c976b5b392a6832f`  
Prompt text SHA-256: `0f9538257bc367d5ae542b835a9efc9d6a8374f1eb1107a38b631b8a8b260efc`

```text
I sometimes develop mild pollen-allergy symptoms that vary from day to day. The unpredictability is wearing me down. What biological processes cause these symptoms, and why can their intensity vary? Please explain the processes rather than recommend steps.
```

#### Neutral · advice

Prompt ID: `prompt_fcf291a1a763545c815d`  
Prompt text SHA-256: `1e561100486cf447bcfa229552e06f4e56d626a148612fefbb9155c1157f1d28`

```text
I sometimes develop mild pollen-allergy symptoms that vary from day to day. What practical steps can I try to reduce routine pollen exposure and make mild allergy days easier to manage?
```

#### Emotional · advice

Prompt ID: `prompt_73b8d5e011a68e0b2bb8`  
Prompt text SHA-256: `95e84894736a237dadc999936728d1ff02446f179d320c8e6a244b22ff8c6240`

```text
I sometimes develop mild pollen-allergy symptoms that vary from day to day. The unpredictability is wearing me down. What practical steps can I try to reduce routine pollen exposure and make mild allergy days easier to manage?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 36. `eval_health_sleep_schedule`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_9e4d0c4bb5a1b1ae6c22`  
Prompt text SHA-256: `319dbc81f38dd876c43daf75f3ad0f18491ac1e36948ed2fa68eac5ec7dd9993`

```text
I often sleep much later on weekends than on weekdays and feel sluggish on Monday. How can the schedule shift affect the circadian rhythm? Please explain the mechanism rather than recommend a schedule.
```

#### Emotional · information

Prompt ID: `prompt_c0a1a3b4e551b8c2e457`  
Prompt text SHA-256: `c64f4091a9fb5990e047e4837c1692edfb4461b28028909473ed870d213a97d2`

```text
I often sleep much later on weekends than on weekdays and feel sluggish on Monday. That Monday sluggishness annoys me. How can the schedule shift affect the circadian rhythm? Please explain the mechanism rather than recommend a schedule.
```

#### Neutral · advice

Prompt ID: `prompt_2561f7625ebbb8aa739d`  
Prompt text SHA-256: `d09428e9a7743b928902cb34eae51760b0d02c0870153e908dea907f76fe4ea1`

```text
I often sleep much later on weekends than on weekdays and feel sluggish on Monday. What gradual scheduling habits can I try to make my sleep times more consistent?
```

#### Emotional · advice

Prompt ID: `prompt_708afa4be24813868259`  
Prompt text SHA-256: `24d28325642d0c30ae0e5a6fddfdf0b722c3b91f782542a09da56cecd83fdd8f`

```text
I often sleep much later on weekends than on weekdays and feel sluggish on Monday. That Monday sluggishness annoys me. What gradual scheduling habits can I try to make my sleep times more consistent?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 37. `eval_health_tension_headache`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_317d6fb869403b32a5f8`  
Prompt text SHA-256: `df73a3186a3383d38ba408ba143fb91b3248bfb7796e3560e9594bf20e7fd736`

```text
I occasionally get a mild, tight headache during a long workday. What factors are thought to contribute to this kind of tension-type headache? Please explain them rather than recommend actions.
```

#### Emotional · information

Prompt ID: `prompt_a2b8246545141a1a4cd2`  
Prompt text SHA-256: `2f94de8b7237be9ed4d7672ee493c902f6f26e4a82d28ed7488e078946a02257`

```text
I occasionally get a mild, tight headache during a long workday. It makes me irritable. What factors are thought to contribute to this kind of tension-type headache? Please explain them rather than recommend actions.
```

#### Neutral · advice

Prompt ID: `prompt_56a21ac5315d1cbf7b58`  
Prompt text SHA-256: `78c77615c227e013260d2d626458c6e13755c74e03f638edf1cd69b16a01b90c`

```text
I occasionally get a mild, tight headache during a long workday. What low-risk habits can I try to prevent or ease it?
```

#### Emotional · advice

Prompt ID: `prompt_2bfe0669b76b486998a7`  
Prompt text SHA-256: `bd0738df3ce2028a753d67b571f04e169303e026610b4a664b80d66f13281417`

```text
I occasionally get a mild, tight headache during a long workday. It makes me irritable. What low-risk habits can I try to prevent or ease it?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 38. `eval_health_tooth_sensitivity`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_a924b6698f7de1df5513`  
Prompt text SHA-256: `18eeff399143276ac1590483538e2926c1c8becfeec2ff8a35243a0b56ad57ca`

```text
One tooth sometimes feels briefly sensitive to cold food or drinks. Why can that happen, and which tooth structures transmit the sensation? Please provide general information rather than care advice.
```

#### Emotional · information

Prompt ID: `prompt_09d05bbdb237b33f3b94`  
Prompt text SHA-256: `557e239abde2b85d4bac6fcdf17556a6204a1b2f2cd11a649549cc288febd154`

```text
One tooth sometimes feels briefly sensitive to cold food or drinks. The sensation makes me uneasy. Why can that happen, and which tooth structures transmit the sensation? Please provide general information rather than care advice.
```

#### Neutral · advice

Prompt ID: `prompt_211392db87dea20cb1d3`  
Prompt text SHA-256: `b40203318d7cc0ccfac2ca56a85996fe31bee1ca764133e1d4bce0016edb513c`

```text
One tooth sometimes feels briefly sensitive to cold food or drinks. What routine oral-care steps can I use while arranging a normal dental checkup if it persists?
```

#### Emotional · advice

Prompt ID: `prompt_2d6077ea6492556b9459`  
Prompt text SHA-256: `1f7f2ea436e1c45c6bec856b46bc0c0e0a2eca073ad6039c7c0662cbc4e60f7b`

```text
One tooth sometimes feels briefly sensitive to cold food or drinks. The sensation makes me uneasy. What routine oral-care steps can I use while arranging a normal dental checkup if it persists?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 39. `eval_health_travel_constipation`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_92421b5838de399b3752`  
Prompt text SHA-256: `0a643d892e75df1ee0cb1051cbf46861be101a9c6bf88188594ccb04d3d074cd`

```text
I sometimes experience temporary constipation when my routine changes during travel. Why can changes in schedule, diet, movement, and hydration contribute? Please explain rather than recommend treatment.
```

#### Emotional · information

Prompt ID: `prompt_efe35a8b9bc74f39f984`  
Prompt text SHA-256: `2e6c113eb2bc1187561ad01abf679f9b7cf3df12c187267955b4c9bd201ceb9e`

```text
I sometimes experience temporary constipation when my routine changes during travel. It leaves me uncomfortable and annoyed. Why can changes in schedule, diet, movement, and hydration contribute? Please explain rather than recommend treatment.
```

#### Neutral · advice

Prompt ID: `prompt_b113100267e8b17fa5f8`  
Prompt text SHA-256: `ca24b592490c6c071fd61f69f6be7d95b467289c637e6a30d2e180f648ea266b`

```text
I sometimes experience temporary constipation when my routine changes during travel. What routine habits can I try when my travel schedule changes?
```

#### Emotional · advice

Prompt ID: `prompt_3ee1ca5008adb7a3b9f1`  
Prompt text SHA-256: `8d7945a212791f2544936ea9e77b758979f54ce5477490644933fcbe21a14bd3`

```text
I sometimes experience temporary constipation when my routine changes during travel. It leaves me uncomfortable and annoyed. What routine habits can I try when my travel schedule changes?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

### 40. `eval_health_winter_skin`

Domain: **health**

#### Neutral · information

Prompt ID: `prompt_03980e22a912b209f8e2`  
Prompt text SHA-256: `5035c2bb07f7dfac028dfd1729eed69818b2df85eff820b3bc736f75468a11cb`

```text
My skin can feel mildly dry during cold weather and indoor heating. Why do cold outdoor air and indoor heating contribute to this? Please explain the skin-barrier process rather than recommend care.
```

#### Emotional · information

Prompt ID: `prompt_bd51db9f183a3ce09949`  
Prompt text SHA-256: `2142d9d9a2d57d4d3113178cc332345811cd984d3b8d9dc72354197683e0afed`

```text
My skin can feel mildly dry during cold weather and indoor heating. The tight feeling bothers me. Why do cold outdoor air and indoor heating contribute to this? Please explain the skin-barrier process rather than recommend care.
```

#### Neutral · advice

Prompt ID: `prompt_3a4870ea1cf4d98b054d`  
Prompt text SHA-256: `77aa522dc6631a89eab6e1121212d060d67d366f93ef6cf706f5a722e43b4797`

```text
My skin can feel mildly dry during cold weather and indoor heating. What basic routine can I try to protect my skin in those conditions?
```

#### Emotional · advice

Prompt ID: `prompt_76c17b16abcd86ba8f57`  
Prompt text SHA-256: `b9bbf4b3b2b6eb83e4ec8b74dfc7d69771ad1e8cd7f92483b7658c6211dac748`

```text
My skin can feel mildly dry during cold weather and indoor heating. The tight feeling bothers me. What basic routine can I try to protect my skin in those conditions?
```

Review judgments:

- `topic_consistent_across_variants`: **PENDING**
- `emotional_framing_manipulation_clear`: **PENDING**
- `advice_information_manipulation_clear`: **PENDING**
- `information_variants_do_not_request_advice`: **PENDING**
- `approved_for_evaluation`: **PENDING**
- `reviewer_notes`: **BLANK**

