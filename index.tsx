/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Chat, Type } from '@google/genai';

const SYSTEM_INSTRUCTION = `你是位專業的語言學習導師。

# 主要規則
1.  **極度簡潔:** 所有回覆都必須精簡，直達重點。避免贅詞、開場白或過度鼓勵。單次回應長度嚴格限制在 200 Token 以內。
2.  **功能性對話:** 完成初始評估後，淡化導師人設，轉為功能性、直接的對話風格。
3.  **靜默執行:** 直接遵循所有規則，不要在對話中解釋或引用它們。

# 語言使用規則
1.  **預設母語 (繁體中文):** 在首次互動、系統性解釋語法、闡述複雜概念，或當用戶明確要求時，你必須使用「繁體中文」進行溝通。
2.  **目標語言練習:** 在進行日常練習、模擬情境對話，或當用戶開始主動使用目標外語時，你必須「全程使用用戶指定的目標外語」來回應。
3.  **錯誤修正:** 你只修正用戶語言中的「關鍵錯誤」。修正時，先用「目標外語」禮貌地提出建議，然後「必須」緊接著用「繁體中文」以括號簡短解釋錯誤原因和正確用法。例如: (這裡用過去式會更自然，因為...)。

# 輸出格式規則
1.  **重點條列:** 當你提供建議、解釋或修正時，**務必**使用列點式（例如：* 或 -）來呈現。
2.  **零干擾練習:** 在角色扮演或情境模擬期間，**絕對不能**使用任何與情境無關的鼓勵或讚美。練習結束後才能給予總結回饋。

# 核心對話流程
1.  **初始評估 (步驟1/3):** 你的第一項任務是歡迎用戶，並詢問他的「目標學習外語」？
2.  **初始評估 (步驟2/3):** 獲取目標語言後，接著詢問用戶目前的「語言程度」？(例如：完全初學者、初級、中級、高級)。
3.  **學習計畫建議 (步驟3/3):**
    *   **判定程度並提出計畫:** 根據用戶的程度，提出一個具體的學習計畫建議，並詢問他是否同意。
    *   **範例 (完全初學者):** 「了解，那我們就從最基礎的發音和常用問候語開始，先打好穩固的基礎。你覺得如何？」
    *   **範例 (中級，但單字量不足):** 「好的。中級程度的話，建議我們可以圍繞一個生活主題（例如旅遊或點餐）來展開對話，並在過程中著重學習相關的新單字和用法。你覺得如何？」
    *   **詢問學習目的:** 在用戶同意計畫後，可以簡短詢問他的「學習目的」(如旅遊、工作)，以便後續微調對話主題。
4.  **開始教學:** 確認計畫後，立即根據計畫展開教學或對話練習。

你的第一則訊息必須是歡迎用戶，並開始初始評估的第一步。`;

const VOCAB_EXTRACTION_PROMPT = `You are an intelligent vocabulary assistant for a language learner whose native language is Traditional Chinese.

Analyze the user's message(s) below and extract up to 5 key vocabulary items (words or short phrases) that are valuable for learning. The messages may be separated by "---".

User's message(s):
"{USER_MESSAGE}"

# Instructions:
1.  **Language Check:** First, determine the primary language of the user's message(s). **If the messages are predominantly in Traditional Chinese, you MUST return an empty array: \`[]\`.** Only extract vocabulary from the user's target language (i.e., not Chinese).

2.  **Identify Key Vocabulary:**
    *   Pinpoint non-trivial words or short, useful phrases (generally 2-4 words long) from the target language.
    *   Ignore basic greetings (e.g., "hello"), pronouns (e.g., "I", "you"), and very common verbs (e.g., "is", "go") unless they are part of a specific, meaningful phrasal verb or idiom.
    *   Focus on words that expand the user's expressive range.

3.  **Translate and Explain:** For each identified item, provide the following fields:
    *   \`term\`: The original word or phrase from the user's message.
    *   \`language\`: The language of the term (e.g., "English", "Japanese", "French").
    *   \`meaning\`: The precise translation and meaning in **Traditional Chinese**.
    *   \`usage\`: A simple, clear example of how to use the word, also written in **Traditional Chinese**.

4.  **Format:** Your entire response must be a valid JSON array of objects.

5.  **Empty Result:** If you find no new or valuable vocabulary worth saving in the messages (or if the messages are in Chinese), you MUST return an empty array: \`[]\`.

# Example (User's target language is English):
User's message(s): "I think this is a spectacular view, it's absolutely breathtaking.---I want to order a coffee, and perhaps a croissant."
Expected JSON Output:
[
  {
    "term": "spectacular view",
    "language": "English",
    "meaning": "壯觀的景色",
    "usage": "這座山頂有著壯觀的景色。"
  },
  {
    "term": "breathtaking",
    "language": "English",
    "meaning": "令人屏息的",
    "usage": "日出的美麗真是令人屏息。"
  },
  {
    "term": "croissant",
    "language": "French",
    "meaning": "可頌麵包",
    "usage": "我早餐喜歡吃可頌麵包配咖啡。"
  }
]

# Example (User message in native language):
User's message(s): "你好，今天天氣真好。"
Expected JSON Output:
[]
`;

// --- TYPES ---
type Proficiency = 'new' | 'learning' | 'mastered';

interface VocabularyItem {
  term: string;
  meaning: string;
  usage: string;
  language: string;
  addedDate: number; // timestamp
  proficiency: Proficiency;
}

// --- DOM ELEMENTS ---
const chatContainer = document.getElementById('chat-container') as HTMLElement;
const chatForm = document.getElementById('chat-form') as HTMLFormElement;
const chatInput = document.getElementById('chat-input') as HTMLInputElement;
const loader = document.getElementById('loader') as HTMLElement;
const sendButton = chatForm.querySelector('button[type="submit"]') as HTMLButtonElement;

const dictionaryBtn = document.getElementById('dictionary-btn') as HTMLButtonElement;
const dictionaryModal = document.getElementById('dictionary-modal') as HTMLElement;
const closeDictionaryBtn = document.getElementById('close-dictionary-btn') as HTMLButtonElement;
const dictionaryForm = document.getElementById('dictionary-form') as HTMLFormElement;
const dictionaryInput = document.getElementById('dictionary-input') as HTMLInputElement;
const dictionaryResults = document.getElementById('dictionary-results') as HTMLElement;
const dictionaryLoader = document.getElementById('dictionary-loader') as HTMLElement;

const vocabularyBtn = document.getElementById('vocabulary-btn') as HTMLButtonElement;
const vocabularyModal = document.getElementById('vocabulary-modal') as HTMLElement;
const closeVocabularyBtn = document.getElementById('close-vocabulary-btn') as HTMLButtonElement;
const vocabStats = document.getElementById('vocab-stats') as HTMLElement;
const sortVocabularySelect = document.getElementById('sort-vocabulary') as HTMLSelectElement;
const languageFilterSelect = document.getElementById('language-filter') as HTMLSelectElement;
const typeFilterContainer = document.getElementById('type-filter') as HTMLElement;
const searchVocabularyInput = document.getElementById('search-vocabulary-input') as HTMLInputElement;
const vocabularyListEl = document.getElementById('vocabulary-list') as HTMLElement;

// --- STATE ---
let chat: Chat;
let ai: GoogleGenAI;
let vocabularyList: VocabularyItem[] = [];
let userMessageBuffer: string[] = [];

// --- CHAT LOGIC ---
function addMessage(sender: 'user' | 'tutor', message: string) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('message', sender);
  messageElement.textContent = message;
  chatContainer.appendChild(messageElement);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function setLoading(isLoading: boolean) {
  if (isLoading) {
    loader.classList.remove('hidden');
    chatInput.disabled = true;
    sendButton.disabled = true;
    chatContainer.scrollTop = chatContainer.scrollHeight;
  } else {
    loader.classList.add('hidden');
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
  }
}

async function sendChatMessage(message: string) {
    if (!message) return;

    addMessage('user', message);
    setLoading(true);

    try {
        const response = await chat.sendMessage({ message: message });
        addMessage('tutor', response.text);
    } catch (error) {
        console.error('Error sending message:', error);
        let errorMessage = '抱歉，我遇到了一些問題，請再試一次。';
        if (error && error.toString().includes('429')) {
            errorMessage = '抱歉，請求頻率過高，請稍候片刻再試。如果您使用的是免費方案，這可能是達到了用量限制。';
        }
        addMessage('tutor', errorMessage);
    } finally {
        setLoading(false);
    }
}


async function handleFormSubmit(event: SubmitEvent) {
  event.preventDefault();
  const userInput = chatInput.value.trim();
  if (!userInput) return;
  chatInput.value = '';

  // Batch vocabulary extraction
  userMessageBuffer.push(userInput);
  if (userMessageBuffer.length >= 3) {
      const combinedMessages = userMessageBuffer.join('\n---\n');
      userMessageBuffer = []; // Clear buffer
      // Delay vocabulary extraction to avoid rate limiting
      setTimeout(() => extractAndSaveVocabulary(combinedMessages), 1500);
  }
  
  // Handle the main chat flow
  await sendChatMessage(userInput);
}

// --- DICTIONARY LOGIC ---
async function handleDictionarySearch(event: SubmitEvent) {
  event.preventDefault();
  const query = dictionaryInput.value.trim();
  if (!query) return;

  dictionaryLoader.classList.remove('hidden');
  dictionaryResults.innerHTML = '';

  const DICTIONARY_PROMPT = `You are a highly intelligent multilingual dictionary assistant for a language learner whose native language is Traditional Chinese.

The user has submitted the following word or phrase for lookup: "${query}"

Your task is to analyze the query and provide contextually relevant results based on its language.

**Instructions:**

1.  **Identify the query's language.** Is it Traditional Chinese or another language (the user's target language)?
2.  **Follow the appropriate output logic:**

    *   **Case A: If the query is in Traditional Chinese (the user's native language):**
        *   Your goal is to provide translations into their likely target language (e.g., English).
        *   Return 3-5 suggestions.
        *   For each suggestion, the 'term' should be in the target language. The 'meaning' and 'usage' must be in Traditional Chinese.
        *   Example Query: "不好意思"
        *   Example 'term' result: "Excuse me"

    *   **Case B: If the query is in the user's target language (e.g., English):**
        *   Your goal is to provide the translation in Traditional Chinese, along with related words in the target language to expand their vocabulary.
        *   Return 3-5 suggestions.
        *   The **first suggestion** should be the primary translation of the query into Traditional Chinese.
        *   The **subsequent suggestions** should be synonyms or closely related words/phrases **in the target language**.
        *   For all suggestions, the 'meaning' and 'usage' must be in Traditional Chinese, explaining the nuance of each term.
        *   Example Query: "Excuse me"
        *   Example 'term' results: "不好意思", "Sorry", "Pardon me"

Always provide your response in the specified JSON format. All explanations ('meaning' and 'usage') MUST be in Traditional Chinese.`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: DICTIONARY_PROMPT,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              term: { type: Type.STRING },
              meaning: { type: Type.STRING },
              usage: { type: Type.STRING },
            },
            required: ['term', 'meaning', 'usage'],
          },
        },
      },
    });

    const results = JSON.parse(response.text);
    renderDictionaryResults(results);
  } catch (error) {
    console.error('Dictionary search error:', error);
    let errorMessage = '<p>抱歉，查詢時發生錯誤。</p>';
    if (error && error.toString().includes('429')) {
        errorMessage = '<p>抱歉，請求頻率過高，請稍候片刻再試。</p>';
    }
    dictionaryResults.innerHTML = errorMessage;
  } finally {
    dictionaryLoader.classList.add('hidden');
  }
}

function renderDictionaryResults(results: { term: string; meaning: string; usage: string }[]) {
  dictionaryResults.innerHTML = '';
  if (!results || results.length === 0) {
    dictionaryResults.innerHTML = '<p>No results found.</p>';
    return;
  }

  results.forEach((result) => {
    const resultEl = document.createElement('div');
    resultEl.classList.add('dictionary-result');

    resultEl.innerHTML = `
      <div class="dictionary-result-main">
        <div class="result-term">${result.term}</div>
        <div class="result-details">
          <span><strong>含義：</strong> ${result.meaning}</span>
          <span><strong>用法：</strong> ${result.usage}</span>
        </div>
      </div>
    `;

    resultEl.querySelector('.dictionary-result-main')?.addEventListener('click', () => {
      navigator.clipboard.writeText(result.term);
      const termEl = resultEl.querySelector('.result-term');
      if (termEl) {
        const originalText = termEl.textContent;
        termEl.textContent = 'Copied!';
        setTimeout(() => {
          termEl.textContent = originalText;
        }, 1000);
      }
    });

    dictionaryResults.appendChild(resultEl);
  });
}

function openDictionary() {
  dictionaryModal.classList.remove('hidden');
  dictionaryInput.focus();
}

function closeDictionary() {
  dictionaryModal.classList.add('hidden');
  chatInput.focus();
}

// --- VOCABULARY LOGIC ---
async function extractAndSaveVocabulary(message: string) {
  // A simple filter to avoid running on very short or common phrases
  if (message.trim().split(' ').length < 2 && message.length < 10) {
      return;
  }

  const prompt = VOCAB_EXTRACTION_PROMPT.replace('{USER_MESSAGE}', message);

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              term: { type: Type.STRING },
              language: { type: Type.STRING },
              meaning: { type: Type.STRING },
              usage: { type: Type.STRING },
            },
            required: ['term', 'language', 'meaning', 'usage'],
          },
        },
      },
    });

    const results = JSON.parse(response.text) as { term: string; language: string; meaning: string; usage: string }[];
    if (results && results.length > 0) {
        // Add words in reverse order so the first word in the message appears on top of the list
        results.reverse().forEach(word => {
            addWordToVocabulary(word);
        });
    }
  } catch (error) {
    // Fail silently so we don't interrupt the user experience.
    console.error('Vocabulary extraction error:', error);
  }
}

function saveVocabulary() {
  localStorage.setItem('vocabularyList', JSON.stringify(vocabularyList));
}

function loadVocabulary() {
  const saved = localStorage.getItem('vocabularyList');
  if (saved) {
    vocabularyList = JSON.parse(saved);
  }
}

function addWordToVocabulary(word: { term: string; language: string; meaning: string; usage: string }) {
  if (vocabularyList.some(item => item.term.toLowerCase() === word.term.toLowerCase())) return; // Avoid duplicates
  const newItem: VocabularyItem = {
    ...word,
    addedDate: Date.now(),
    proficiency: 'new',
  };
  vocabularyList.unshift(newItem); // Add to the beginning
  saveVocabulary();
  
  // Update view only if it's currently open
  if (!vocabularyModal.classList.contains('hidden')) {
    populateLanguageFilter();
    renderVocabularyList();
  }
}

function updateProficiency(term: string, proficiency: Proficiency) {
  const item = vocabularyList.find(i => i.term === term);
  if (item) {
    item.proficiency = proficiency;
    saveVocabulary();
    renderVocabularyList();
  }
}

async function practiceWord(term: string) {
    closeVocabulary();
    const prompt = `請針對 "${term}" 這個單字或片語，給我一個練習。`;
    await sendChatMessage(prompt);
}

function populateLanguageFilter() {
    const languages = [...new Set(vocabularyList.map(item => item.language))];
    const currentSelection = languageFilterSelect.value;
    languageFilterSelect.innerHTML = '<option value="all">全部語言</option>';
    languages.sort().forEach(lang => {
        const option = document.createElement('option');
        option.value = lang;
        option.textContent = lang;
        languageFilterSelect.appendChild(option);
    });
    // Preserve selection if possible
    if (languages.includes(currentSelection)) {
        languageFilterSelect.value = currentSelection;
    } else {
        languageFilterSelect.value = 'all';
    }
}

function renderVocabularyList() {
    // 1. Get filter values
    const selectedLanguage = languageFilterSelect.value;
    const selectedType = (document.querySelector('input[name="type-filter"]:checked') as HTMLInputElement)?.value || 'all';
    const searchQuery = searchVocabularyInput.value.toLowerCase().trim();
    const sortBy = sortVocabularySelect.value;

    // 2. Filter list
    const filteredList = vocabularyList.filter(item => {
        // Language filter
        if (selectedLanguage !== 'all' && item.language !== selectedLanguage) {
            return false;
        }
        // Type filter
        const isPhrase = item.term.trim().includes(' ');
        if (selectedType === 'word' && isPhrase) {
            return false;
        }
        if (selectedType === 'phrase' && !isPhrase) {
            return false;
        }
        // Search filter
        if (searchQuery &&
            !item.term.toLowerCase().includes(searchQuery) &&
            !item.meaning.toLowerCase().includes(searchQuery)) {
            return false;
        }
        return true;
    });

    // 3. Update stats
    const today = new Date().setHours(0, 0, 0, 0);
    const todayCount = vocabularyList.filter(item => item.addedDate >= today).length;
    vocabStats.textContent = `今日新學：${todayCount} / 全部：${vocabularyList.length}`;

    // 4. Sort the filtered list
    const sortedList = [...filteredList].sort((a, b) => {
        if (sortBy === 'alphabetical') {
            return a.term.localeCompare(b.term);
        }
        if (sortBy === 'proficiency') {
            const pOrder: Record<Proficiency, number> = { 'new': 0, 'learning': 1, 'mastered': 2 };
            return pOrder[a.proficiency] - pOrder[b.proficiency];
        }
        // Default to date
        return b.addedDate - a.addedDate;
    });

    // 5. Render
    vocabularyListEl.innerHTML = '';
    if (vocabularyList.length === 0) {
        vocabularyListEl.innerHTML = '<p class="empty-list-message">你的學習紀錄是空的。請先在字典中查詢並儲存單字。</p>';
        return;
    }
    if (sortedList.length === 0) {
        vocabularyListEl.innerHTML = '<p class="empty-list-message">找不到符合條件的單字。請調整篩選條件。</p>';
        return;
    }

    sortedList.forEach(item => {
        const itemEl = document.createElement('div');
        itemEl.classList.add('vocabulary-item');
        itemEl.addEventListener('click', () => practiceWord(item.term));

        const dateString = new Date(item.addedDate).toLocaleDateString();

        itemEl.innerHTML = `
            <div class="vocab-main">
                <div class="vocab-term">${item.term}</div>
                <div class="vocab-details">${item.meaning}</div>
            </div>
            <div class="vocab-meta">
                 <select class="proficiency-select" data-term="${item.term}">
                    <option value="new" ${item.proficiency === 'new' ? 'selected' : ''}>新學</option>
                    <option value="learning" ${item.proficiency === 'learning' ? 'selected' : ''}>學習中</option>
                    <option value="mastered" ${item.proficiency === 'mastered' ? 'selected' : ''}>已掌握</option>
                </select>
                <span>${dateString}</span>
            </div>
        `;

        const proficiencySelect = itemEl.querySelector('.proficiency-select') as HTMLSelectElement;
        proficiencySelect.addEventListener('click', (e) => e.stopPropagation()); // Prevent item click
        proficiencySelect.addEventListener('change', () => {
            updateProficiency(item.term, proficiencySelect.value as Proficiency);
        });

        vocabularyListEl.appendChild(itemEl);
    });
}

function openVocabulary() {
  populateLanguageFilter();
  renderVocabularyList();
  vocabularyModal.classList.remove('hidden');
}

function closeVocabulary() {
  vocabularyModal.classList.add('hidden');
  chatInput.focus();
}


// --- INITIALIZATION ---
async function initializeApp() {
  setLoading(true);
  try {
    loadVocabulary();
    ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    chat = ai.chats.create({
      model: 'gemini-2.5-flash',
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
        maxOutputTokens: 200,
        thinkingConfig: { thinkingBudget: 50 },
       },
    });
    
    // Replace API call with a hardcoded initial message based on the system prompt.
    const welcomeMessage = "你好！我是你的專屬語言導師。為了能更有效地幫助你，我想先了解一下你的學習目標。請問，你最想學習的是哪一種語言呢？";
    addMessage('tutor', welcomeMessage);

  } catch (error) {
    console.error('Initialization error:', error);
    let errorMessage = '抱歉，初始化導師時發生錯誤。請檢查您的網路連線或 API 金鑰後，重新整理頁面。';
    if (error && error.toString().includes('429')) {
        errorMessage = '抱歉，請求頻率過高，請稍候片刻再試。如果您使用的是免費方案，這可能是達到了用量限制。請稍後重新整理頁面。';
    }
    addMessage('tutor', errorMessage);
  } finally {
    setLoading(false);
  }
}

function main() {
  // Chat listeners
  chatForm.addEventListener('submit', handleFormSubmit);
  
  // Dictionary listeners
  dictionaryBtn.addEventListener('click', openDictionary);
  closeDictionaryBtn.addEventListener('click', closeDictionary);
  dictionaryForm.addEventListener('submit', handleDictionarySearch);
  dictionaryModal.addEventListener('click', (event) => {
    if (event.target === dictionaryModal) closeDictionary();
  });

  // Vocabulary listeners
  vocabularyBtn.addEventListener('click', openVocabulary);
  closeVocabularyBtn.addEventListener('click', closeVocabulary);
  sortVocabularySelect.addEventListener('change', renderVocabularyList);
  languageFilterSelect.addEventListener('change', renderVocabularyList);
  searchVocabularyInput.addEventListener('input', renderVocabularyList);
  typeFilterContainer.addEventListener('change', renderVocabularyList);
  vocabularyModal.addEventListener('click', (event) => {
    if (event.target === vocabularyModal) closeVocabulary();
  });

  initializeApp();
}

main();