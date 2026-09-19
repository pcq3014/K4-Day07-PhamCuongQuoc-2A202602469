/**
 * UniRAG Frontend Application Logic
 * Interactive RAG Chat, Knowledge Base Explorer, Chunking Lab, and Benchmark Dashboard
 */

// State Management
const state = {
  activeTab: 'chat',
  documents: [],
  filteredDocuments: [],
  selectedSchool: 'all',
  theme: 'light',
  currentPrompt: '',
  isQuerying: false,
};

// DOM Elements
const elements = {
  // Tabs
  navTabs: document.querySelectorAll('.nav-tab'),
  tabViews: document.querySelectorAll('.tab-view'),
  docCountBadge: document.getElementById('doc-count-badge'),
  backendStatusText: document.getElementById('backend-status-text'),
  sidebarCorpusStat: document.getElementById('sidebar-corpus-stat'),
  currentFilterIndicator: document.getElementById('current-filter-indicator'),

  // Theme & Modals
  themeToggleBtn: document.getElementById('btn-theme-toggle'),
  themeIcon: document.getElementById('theme-icon'),
  guideBtn: document.getElementById('btn-guide-modal'),
  guideModal: document.getElementById('guide-modal'),
  closeGuideBtns: [document.getElementById('btn-close-guide'), document.getElementById('btn-close-guide-bottom')],

  // Tab 1: Chat
  chatForm: document.getElementById('chat-form'),
  chatInput: document.getElementById('chat-input'),
  chatMessages: document.getElementById('chat-messages'),
  filterAudience: document.getElementById('filter-audience'),
  sliderTopk: document.getElementById('slider-topk'),
  valTopk: document.getElementById('val-topk'),
  selectStrategy: document.getElementById('select-strategy'),
  selectMode: document.getElementById('select-mode'),
  presetChips: document.querySelectorAll('.preset-chip'),
  btnClearInput: document.getElementById('btn-clear-input'),
  btnClearChat: document.getElementById('btn-clear-chat'),
  btnSend: document.getElementById('btn-send'),

  // Tab 2: Knowledge Base
  docsGrid: document.getElementById('docs-grid'),
  searchDocsInput: document.getElementById('search-docs-input'),
  schoolFilterContainer: document.getElementById('school-filter-container'),
  btnOpenAddDoc: document.getElementById('btn-open-add-doc'),
  addDocModal: document.getElementById('add-doc-modal'),
  btnCloseAddDoc: document.getElementById('btn-close-add-doc'),
  addDocForm: document.getElementById('add-doc-form'),

  // Tab 3: Playground
  presetChunkSample: document.getElementById('preset-chunk-sample'),
  chunkInputText: document.getElementById('chunk-input-text'),
  chunkCharCounter: document.getElementById('chunk-char-counter'),
  chunkParamSize: document.getElementById('chunk-param-size'),
  chunkParamOverlap: document.getElementById('chunk-param-overlap'),
  btnRunChunkComparison: document.getElementById('btn-run-chunk-comparison'),
  chunkingResultsGrid: document.getElementById('chunking-results-grid'),

  presetSimilarityPairs: document.getElementById('preset-similarity-pairs'),
  simText1: document.getElementById('sim-text-1'),
  simText2: document.getElementById('sim-text-2'),
  btnCalcSimilarity: document.getElementById('btn-calc-similarity'),
  simScoreVal: document.getElementById('sim-score-val'),
  simMeterFill: document.getElementById('sim-meter-fill'),
  simExplanationText: document.getElementById('sim-explanation-text'),

  // Tab 4: Benchmark
  benchSelectStrategy: document.getElementById('bench-select-strategy'),
  btnRunBenchmark: document.getElementById('btn-run-benchmark'),
  benchTotalScore: document.getElementById('bench-total-score'),
  benchPercentage: document.getElementById('bench-percentage'),
  benchQueriesList: document.getElementById('bench-queries-list'),

  // Modals
  docModal: document.getElementById('doc-modal'),
  btnCloseDocModal: document.getElementById('btn-close-doc-modal'),
  btnCloseDocModalBottom: document.getElementById('btn-close-doc-modal-bottom'),
  modalDocSchool: document.getElementById('modal-doc-school'),
  modalDocTitle: document.getElementById('modal-doc-title'),
  modalDocMeta: document.getElementById('modal-doc-meta'),
  modalDocRaw: document.getElementById('modal-doc-raw'),
  modalDocChunks: document.getElementById('modal-doc-chunks'),
  modalDocSourceLink: document.getElementById('modal-doc-source-link'),

  promptModal: document.getElementById('prompt-modal'),
  btnClosePromptModal: document.getElementById('btn-close-prompt-modal'),
  btnClosePromptModalBottom: document.getElementById('btn-close-prompt-modal-bottom'),
  modalPromptContent: document.getElementById('modal-prompt-content'),
  btnCopyPrompt: document.getElementById('btn-copy-prompt'),

  toastContainer: document.getElementById('toast-container'),
};

// ================= INITIALIZATION =================
function initApp() {
  initTabs();
  initTheme();
  initChatEvents();
  initKnowledgeBaseEvents();
  initPlaygroundEvents();
  initBenchmarkEvents();
  initModals();

  // Load initial backend info & documents
  fetchStatus();
  fetchDocuments();
}

// Phần khởi động nằm ở CUỐI file (xem bootstrap ở dòng cuối cùng).
// Không gọi initApp() tại đây: initPlaygroundEvents() đọc `SAMPLE_PRESETS`,
// vốn là một `const` khai báo phía dưới. Khi trình duyệt đã parse xong DOM
// (`readyState !== 'loading'`), lệnh gọi ngay tại đây sẽ chạm vào biến đó
// trước khi nó được khởi tạo và ném ReferenceError (temporal dead zone),
// khiến initBenchmarkEvents() và initModals() không bao giờ được gắn.

// Toast Helper
function showToast(message, duration = 3000) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerText = message;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// Markdown Formatter (Lightweight & Safe)
function renderMarkdown(text) {
  if (!text) return '';
  let escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Citations [1], [2] -> interactive badges
  escaped = escaped.replace(/\[(\d+)\]/g, '<span class="citation-badge" data-cite="$1">[$1]</span>');

  // Bold **text**
  escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Inline code `code`
  escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Line breaks to paragraphs
  const paragraphs = escaped.split('\n\n').map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`);
  return paragraphs.join('');
}

// ================= TABS NAVIGATION =================
function initTabs() {
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = tab.dataset.tab;
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.activeTab = tabId;
  elements.navTabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
  elements.tabViews.forEach(v => v.classList.toggle('active', v.id === `view-${tabId}`));

  if (tabId === 'knowledge' && state.documents.length === 0) {
    fetchDocuments();
  } else if (tabId === 'playground') {
    if (!elements.chunkingResultsGrid.querySelector('.strategy-stat-card')) {
      elements.btnRunChunkComparison.click();
    }
  } else if (tabId === 'benchmark') {
    if (!document.querySelector('.strategies-comparison-table')) {
      elements.btnRunBenchmark.click();
    }
  }
}

// ================= THEME TOGGLE =================
function initTheme() {
  elements.themeToggleBtn.addEventListener('click', () => {
    const isDark = document.body.classList.contains('dark-theme');
    if (isDark) {
      document.body.classList.remove('dark-theme');
      document.body.classList.add('light-theme');
      elements.themeIcon.innerHTML = '<svg class="icon" aria-hidden="true"><use href="#i-sun"/></svg>';
      state.theme = 'light';
    } else {
      document.body.classList.remove('light-theme');
      document.body.classList.add('dark-theme');
      elements.themeIcon.innerHTML = '<svg class="icon" aria-hidden="true"><use href="#i-moon"/></svg>';
      state.theme = 'dark';
    }
  });
}

// ================= API FETCH STATUS =================
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('API error');
    const data = await res.json();
    
    if (elements.backendStatusText) {
      elements.backendStatusText.innerText = `RAG Online • ${data.topic}`;
    }
    if (elements.docCountBadge) {
      elements.docCountBadge.innerText = data.corpus_doc_count;
    }
    if (elements.sidebarCorpusStat) {
      elements.sidebarCorpusStat.innerText = `${data.corpus_doc_count} tài liệu • K4-L3A`;
    }
  } catch (err) {
    if (elements.backendStatusText) {
      elements.backendStatusText.innerText = 'RAG Server Đang Khởi Động';
    }
  }
}

// ================= TAB 1: CHAT LOGIC =================
function initChatEvents() {
  // Top-K Slider
  elements.sliderTopk.addEventListener('input', (e) => {
    elements.valTopk.innerText = e.target.value;
  });

  // Audience Filter Change
  elements.filterAudience.addEventListener('change', (e) => {
    const textMap = {
      all: 'Tất cả độc giả (all)',
      student: 'Sinh viên (student)',
      faculty: 'Giảng viên / Cán bộ (faculty)',
      staff: 'Nhân viên (staff)',
    };
    elements.currentFilterIndicator.innerText = textMap[e.target.value] || e.target.value;
  });

  // Input autosize & clear
  elements.chatInput.addEventListener('input', () => {
    elements.chatInput.style.height = 'auto';
    elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 160) + 'px';
    elements.btnClearInput.style.display = elements.chatInput.value.length > 0 ? 'block' : 'none';
  });

  elements.btnClearInput.addEventListener('click', () => {
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    elements.btnClearInput.style.display = 'none';
    elements.chatInput.focus();
  });

  // Enter to send (Shift+Enter for newline)
  elements.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  // Preset Chips Click
  elements.presetChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.preventDefault();
      elements.presetChips.forEach(c => c.classList.remove('active-chip'));
      chip.classList.add('active-chip');

      const query = chip.dataset.query;
      const audience = chip.dataset.audience;
      if (audience) {
        elements.filterAudience.value = audience;
        elements.filterAudience.dispatchEvent(new Event('change'));
      }
      if (query) {
        sendChatMessage(query);
      }
    });
  });

  // Clear Chat History Button
  if (elements.btnClearChat) {
    elements.btnClearChat.addEventListener('click', (e) => {
      e.preventDefault();
      elements.chatMessages.innerHTML = `
        <div class="message-row assistant">
          <div class="avatar assistant-avatar"><svg class="icon" aria-hidden="true"><use href="#i-logo"/></svg></div>
          <div class="message-bubble">
            <div class="message-meta">
              <span class="sender-name">UniRAG Assistant</span>
              <span class="meta-tag">Hệ Thống RAG Cơ Sở Tri Thức</span>
            </div>
            <div class="message-text">
              <p>Xin chào! Tôi là trợ lý tra cứu cơ sở tri thức về <strong>Quy chế & Dịch vụ Đại học</strong> (ĐH GTVT, HV Ngoại Giao, ĐH Thương Mại, ĐH Công Nghiệp Hà Nội...).</p>
              <p>Mỗi câu trả lời của tôi đều được <strong>truy xuất và trích dẫn trực tiếp [1], [2]</strong> từ tài liệu quy định thực tế của các trường, tuyệt đối không bịa đặt thông tin.</p>
              <div class="welcome-quick-actions">
                <span>Bạn có thể chọn nhanh một câu hỏi bên trái hoặc gõ câu hỏi bất kỳ bên dưới!</span>
              </div>
            </div>
          </div>
        </div>
      `;
      elements.presetChips.forEach(c => c.classList.remove('active-chip'));
      elements.chatInput.value = '';
      elements.chatInput.style.height = 'auto';
      if (elements.btnClearInput) elements.btnClearInput.style.display = 'none';
      state.isQuerying = false;
      if (elements.btnSend) {
        elements.btnSend.disabled = false;
        elements.btnSend.innerHTML = '<span class="btn-send-icon"><svg class="icon" aria-hidden="true"><use href="#i-send"/></svg></span><span class="btn-send-label">Gửi</span>';
      }
      elements.chatInput.disabled = false;
      elements.chatInput.focus();
      showToast('Đã làm mới phiên hỏi đáp');
    });
  }

  // Submit Query Form
  elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendChatMessage();
  });

  // Direct Send Button Click
  if (elements.btnSend) {
    elements.btnSend.addEventListener('click', (e) => {
      e.preventDefault();
      sendChatMessage();
    });
  }
}

async function sendChatMessage(customQuestion = null) {
  const question = (customQuestion !== null ? customQuestion : elements.chatInput.value).trim();
  if (!question || state.isQuerying) return;

  // Append User Message
  appendUserMessage(question);
  elements.chatInput.value = '';
  elements.chatInput.style.height = 'auto';
  if (elements.btnClearInput) elements.btnClearInput.style.display = 'none';

  // Disable button and input during request
  if (elements.btnSend) {
    elements.btnSend.disabled = true;
    elements.btnSend.innerHTML = '<span class="spinner" style="width: 15px; height: 15px; border-width: 2px;"></span>';
  }
  elements.chatInput.disabled = true;

  // Show Assistant Loading Message
  const loadingMessageId = appendLoadingMessage();
  state.isQuerying = true;

  try {
    const payload = {
      question: question,
      top_k: parseInt(elements.sliderTopk.value, 10),
      audience: elements.filterAudience.value,
      strategy: elements.selectStrategy.value,
      mode: elements.selectMode.value,
    };

    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error('Không thể kết nối tới server RAG');
    const data = await res.json();

    // Replace loading bubble with assistant answer
    updateAssistantMessage(loadingMessageId, data);
  } catch (err) {
    updateAssistantError(loadingMessageId, err.message || 'Lỗi xử lý câu hỏi');
  } finally {
    state.isQuerying = false;
    if (elements.btnSend) {
      elements.btnSend.disabled = false;
      elements.btnSend.innerHTML = '<span class="btn-send-icon"><svg class="icon" aria-hidden="true"><use href="#i-send"/></svg></span><span class="btn-send-label">Gửi</span>';
    }
    elements.chatInput.disabled = false;
    elements.chatInput.focus();
  }
}

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="avatar user-avatar"><svg class="icon" aria-hidden="true"><use href="#i-user"/></svg></div>
    <div class="message-bubble">
      <div class="message-meta">
        <span class="sender-name">Bạn</span>
      </div>
      <div class="message-text">
        <p>${escapeHtml(text)}</p>
      </div>
    </div>
  `;
  elements.chatMessages.appendChild(row);
  scrollToBottom();
}

function appendLoadingMessage() {
  const id = 'msg-' + Date.now();
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.id = id;
  row.innerHTML = `
    <div class="avatar assistant-avatar"><svg class="icon" aria-hidden="true"><use href="#i-logo"/></svg></div>
    <div class="message-bubble">
      <div class="message-meta">
        <span class="sender-name">UniRAG Assistant</span>
        <span class="meta-tag">Đang truy xuất văn bản quy định...</span>
      </div>
      <div class="message-text">
        <div class="typing-dots">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  elements.chatMessages.appendChild(row);
  scrollToBottom();
  return id;
}

function updateAssistantMessage(id, data) {
  const row = document.getElementById(id);
  if (!row) return;

  state.currentPrompt = data.prompt || '';

  const chunksHtml = (data.citations || []).map(chunk => `
    <div class="chunk-item-card" id="chunk-card-${chunk.index}">
      <div class="chunk-card-top">
        <div class="chunk-title-group">
          <span class="chunk-rank">[${chunk.index}]</span>
          <span class="chunk-doc-name">${escapeHtml(chunk.doc_id)}</span>
        </div>
        <div class="chunk-meta-badges">
          <span class="score-badge" title="Cosine Similarity Score">Score: ${chunk.score}</span>
          <span class="audience-badge ${chunk.audience}">${chunk.audience}</span>
        </div>
      </div>
      <div class="chunk-body-text">${escapeHtml(chunk.content)}</div>
      ${chunk.source_url && chunk.source_url !== '#' ? `
        <div class="chunk-card-footer">
          <a href="${chunk.source_url}" target="_blank" rel="noopener" class="link-source">
            <span>Trang nguồn đại học</span><svg class="icon" aria-hidden="true"><use href="#i-external"/></svg>
          </a>
        </div>
      ` : ''}
    </div>
  `).join('');

  row.innerHTML = `
    <div class="avatar assistant-avatar"><svg class="icon" aria-hidden="true"><use href="#i-logo"/></svg></div>
    <div class="message-bubble">
      <div class="message-meta">
        <span class="sender-name">UniRAG Assistant</span>
        <span class="meta-tag">Trích xuất ${data.citations?.length || 0} Chunks • ${data.execution_time_ms}ms</span>
      </div>
      <div class="message-text">
        ${renderMarkdown(data.answer)}
      </div>

      ${data.citations && data.citations.length > 0 ? `
        <div class="retrieved-chunks-section open" id="chunks-sec-${id}">
          <button class="chunks-section-toggle" onclick="toggleChunks('${id}')">
            <svg class="icon" aria-hidden="true"><use href="#i-library"/></svg><span>Ngữ cảnh đối chiếu (${data.citations.length} đoạn trích nguồn)</span>
            <span class="toggle-icon">▼</span>
          </button>
          <div class="chunks-list-container">
            ${chunksHtml}
          </div>
        </div>
      ` : ''}

      <div class="message-actions-bar">
        <span>Chiến lược: <strong>${data.strategy}</strong></span>
        ${data.prompt ? `
          <button class="btn-inspect-prompt" onclick="openPromptModal()">
            <svg class="icon" aria-hidden="true"><use href="#i-search"/></svg><span>Xem Prompt Gốc</span>
          </button>
        ` : ''}
      </div>
    </div>
  `;

  // Attach click listener for citations inside this message
  row.querySelectorAll('.citation-badge').forEach(badge => {
    badge.addEventListener('click', (e) => {
      const citeIdx = badge.dataset.cite;
      const targetChunk = row.querySelector(`#chunk-card-${citeIdx}`);
      if (targetChunk) {
        row.querySelectorAll('.chunk-item-card').forEach(c => c.classList.remove('highlighted'));
        targetChunk.classList.add('highlighted');
        targetChunk.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  });

  scrollToBottom();
}

function updateAssistantError(id, errorMsg) {
  const row = document.getElementById(id);
  if (!row) return;
  row.innerHTML = `
    <div class="avatar assistant-avatar is-error"><svg class="icon" aria-hidden="true"><use href="#i-alert"/></svg></div>
    <div class="message-bubble" style="border-color: rgba(244,63,94,0.4);">
      <div class="message-meta">
        <span class="sender-name">Lỗi Hệ Thống RAG</span>
      </div>
      <div class="message-text" style="color: var(--rose);">
        <p>${escapeHtml(errorMsg)}</p>
      </div>
    </div>
  `;
  scrollToBottom();
}

window.toggleChunks = function(msgId) {
  const sec = document.getElementById(`chunks-sec-${msgId}`);
  if (sec) {
    sec.classList.toggle('open');
  }
};

function scrollToBottom() {
  if (elements.chatMessages) {
    elements.chatMessages.scrollTo({
      top: elements.chatMessages.scrollHeight,
      behavior: 'smooth'
    });
  }
}

// ================= TAB 2: KNOWLEDGE BASE =================
function initKnowledgeBaseEvents() {
  // Search input
  elements.searchDocsInput.addEventListener('input', (e) => {
    filterDocs(e.target.value, state.selectedSchool);
  });

  // School filter pills
  elements.schoolFilterContainer.addEventListener('click', (e) => {
    const pill = e.target.closest('.filter-pill');
    if (!pill) return;
    elements.schoolFilterContainer.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    state.selectedSchool = pill.dataset.school;
    filterDocs(elements.searchDocsInput.value, state.selectedSchool);
  });

  // Open add document modal
  elements.btnOpenAddDoc.addEventListener('click', () => {
    elements.addDocModal.style.display = 'flex';
  });

  elements.btnCloseAddDoc.addEventListener('click', () => {
    elements.addDocModal.style.display = 'none';
  });

  // Submit Add Document Form
  elements.addDocForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      title: document.getElementById('add-doc-title').value.trim(),
      filename: document.getElementById('add-doc-filename').value.trim(),
      audience: document.getElementById('add-doc-audience').value,
      department: document.getElementById('add-doc-dept').value.trim(),
      source_url: document.getElementById('add-doc-url').value.trim(),
      content: document.getElementById('add-doc-content').value.trim(),
    };

    try {
      const res = await fetch('/api/documents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('Không thể thêm tài liệu');
      showToast('Đã thêm tài liệu mới vào cơ sở tri thức thành công!');
      elements.addDocModal.style.display = 'none';
      elements.addDocForm.reset();
      fetchDocuments();
    } catch (err) {
      showToast('Lỗi: ' + err.message);
    }
  });
}

async function fetchDocuments() {
  try {
    const res = await fetch('/api/documents');
    if (!res.ok) throw new Error('Không thể tải tài liệu');
    state.documents = await res.json();

    // Dynamically update school filter pill counts
    if (elements.schoolFilterContainer) {
      const counts = { all: state.documents.length };
      state.documents.forEach(d => {
        counts[d.school] = (counts[d.school] || 0) + 1;
      });
      elements.schoolFilterContainer.querySelectorAll('.filter-pill').forEach(pill => {
        const sch = pill.dataset.school;
        let baseName = 'Tất cả';
        if (sch === 'all') baseName = 'Tất cả';
        else if (sch.includes('Giao thông')) baseName = 'ĐH GTVT';
        else if (sch.includes('Ngoại giao')) baseName = 'HV Ngoại Giao';
        else if (sch.includes('Thương mại')) baseName = 'ĐH Thương Mại';
        else if (sch.includes('Công nghiệp')) baseName = 'ĐH Công Nghiệp';
        else baseName = sch;

        const count = sch === 'all' ? counts.all : (counts[sch] || 0);
        pill.innerText = `${baseName} (${count})`;
      });
    }

    filterDocs(elements.searchDocsInput.value, state.selectedSchool);
    if (elements.docCountBadge) elements.docCountBadge.innerText = state.documents.length;
  } catch (err) {
    elements.docsGrid.innerHTML = `<div class="empty-state-hint">Lỗi tải tài liệu: ${err.message}</div>`;
  }
}

function filterDocs(searchTerm, school) {
  const term = (searchTerm || '').toLowerCase();
  state.filteredDocuments = state.documents.filter(doc => {
    const matchSchool = (school === 'all') || (doc.school === school);
    const matchSearch = !term ||
      doc.title.toLowerCase().includes(term) ||
      doc.filename.toLowerCase().includes(term) ||
      doc.preview.toLowerCase().includes(term) ||
      doc.department.toLowerCase().includes(term);
    return matchSchool && matchSearch;
  });
  renderDocsGrid();
}

function renderDocsGrid() {
  if (!state.filteredDocuments || state.filteredDocuments.length === 0) {
    elements.docsGrid.innerHTML = `
      <div class="empty-state-hint" style="grid-column: 1 / -1;">
        <span>Không tìm thấy tài liệu phù hợp với bộ lọc tìm kiếm.</span>
      </div>
    `;
    return;
  }

  elements.docsGrid.innerHTML = state.filteredDocuments.map(doc => `
    <div class="doc-card">
      <div class="doc-card-top">
        <div class="doc-school-badge">${escapeHtml(doc.school)}</div>
        <h3 class="doc-card-title">${escapeHtml(doc.title)}</h3>
        <div class="doc-meta-tags">
          <span class="meta-chip audience-badge ${doc.audience}">Audience: ${doc.audience}</span>
          <span class="meta-chip">Dept: ${escapeHtml(doc.department)}</span>
          <span class="meta-chip">${doc.char_count} chars</span>
        </div>
        <p class="doc-card-preview">${escapeHtml(doc.preview)}</p>
      </div>
      <div class="doc-card-actions">
        <button class="btn btn-secondary" onclick="viewDocDetails('${doc.id}')">
          <svg class="icon" aria-hidden="true"><use href="#i-file"/></svg><span>Xem chi tiết &amp; chunks</span>
        </button>
        ${doc.source_url && doc.source_url !== '#' ? `
          <a href="${doc.source_url}" target="_blank" rel="noopener" class="link-source">
            <span>Nguồn</span><svg class="icon" aria-hidden="true"><use href="#i-external"/></svg>
          </a>
        ` : ''}
      </div>
    </div>
  `).join('');
}

// ================= BEAUTIFUL MARKDOWN RENDERER =================
function renderFullMarkdown(rawText) {
  if (!rawText) return '<p class="text-muted">Không có nội dung</p>';

  // 1. Strip YAML Frontmatter completely from rendered view
  let text = rawText.replace(/^---[\s\S]*?---\s*/, '').trim();

  let lines = text.split('\n');
  let html = [];
  let inTable = false;
  let tableHeaders = [];
  let tableRows = [];
  let inList = false;
  let listType = 'ul';

  const flushTable = () => {
    if (!inTable) return;
    let tHtml = '<table class="rendered-table"><thead><tr>';
    tableHeaders.forEach(h => {
      tHtml += `<th>${formatInline(h.trim())}</th>`;
    });
    tHtml += '</tr></thead><tbody>';
    tableRows.forEach(row => {
      tHtml += '<tr>';
      row.forEach(cell => {
        tHtml += `<td>${formatInline(cell.trim())}</td>`;
      });
      tHtml += '</tr>';
    });
    tHtml += '</tbody></table>';
    html.push(tHtml);
    inTable = false;
    tableHeaders = [];
    tableRows = [];
  };

  const flushList = () => {
    if (!inList) return;
    html.push(`</${listType}>`);
    inList = false;
  };

  const formatInline = (str) => {
    return str
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\[(\d+)\]/g, '<span class="citation-badge">[$1]</span>');
  };

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();

    // Table detection: line starts and ends with |
    if (line.startsWith('|') && line.endsWith('|')) {
      flushList();
      let cells = line.split('|').slice(1, -1);
      // Skip separator line (| --- | --- |)
      if (cells.every(c => c.trim().match(/^:?-+:?$/))) {
        continue;
      }
      if (!inTable) {
        inTable = true;
        tableHeaders = cells;
      } else {
        tableRows.push(cells);
      }
      continue;
    } else {
      flushTable();
    }

    if (!line) {
      flushList();
      continue;
    }

    // Headings
    if (line.startsWith('### ')) {
      flushList();
      html.push(`<h3>${formatInline(line.slice(4))}</h3>`);
      continue;
    }
    if (line.startsWith('## ')) {
      flushList();
      html.push(`<h2><span style="color: var(--primary);">§</span> ${formatInline(line.slice(3))}</h2>`);
      continue;
    }
    if (line.startsWith('# ')) {
      flushList();
      html.push(`<h1>${formatInline(line.slice(2))}</h1>`);
      continue;
    }

    // Blockquote
    if (line.startsWith('> ')) {
      flushList();
      html.push(`<blockquote>${formatInline(line.slice(2))}</blockquote>`);
      continue;
    }

    // Horizontal Rule
    if (line === '---' || line === '***' || line === '___') {
      flushList();
      html.push('<hr style="border: none; border-top: 1px solid var(--border-subtle); margin: 1.5rem 0;">');
      continue;
    }

    // Unordered List
    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList || listType !== 'ul') {
        flushList();
        inList = true;
        listType = 'ul';
        html.push('<ul>');
      }
      html.push(`<li>${formatInline(line.slice(2))}</li>`);
      continue;
    }

    // Ordered List
    let numMatch = line.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      if (!inList || listType !== 'ol') {
        flushList();
        inList = true;
        listType = 'ol';
        html.push('<ol>');
      }
      html.push(`<li>${formatInline(numMatch[2])}</li>`);
      continue;
    }

    // Normal paragraph
    flushList();
    html.push(`<p>${formatInline(line)}</p>`);
  }

  flushTable();
  flushList();

  return html.join('');
}

window.viewDocDetails = function(docId) {
  const doc = state.documents.find(d => d.id === docId);
  if (!doc) return;

  const audienceLabels = {
    student: 'Sinh viên / Học viên',
    faculty: 'Cán bộ / Giảng viên',
    staff: 'Nhân viên thư viện',
    all: 'Toàn thể nhà trường',
  };

  elements.modalDocSchool.innerText = doc.school;
  elements.modalDocTitle.innerText = doc.title;
  elements.modalDocSourceLink.href = doc.source_url || '#';

  // 1. 4-Tile Metadata Grid
  elements.modalDocMeta.innerHTML = `
    <div class="doc-meta-tile">
      <div class="meta-tile-label"><svg class="icon" aria-hidden="true"><use href="#i-users"/></svg>Đối tượng áp dụng</div>
      <div class="meta-tile-value">
        <span class="audience-badge ${doc.audience}">${audienceLabels[doc.audience] || doc.audience}</span>
      </div>
    </div>
    <div class="doc-meta-tile">
      <div class="meta-tile-label"><svg class="icon" aria-hidden="true"><use href="#i-building"/></svg>Đơn vị ban hành</div>
      <div class="meta-tile-value" title="${escapeHtml(doc.department)}">${escapeHtml(doc.department)}</div>
    </div>
    <div class="doc-meta-tile">
      <div class="meta-tile-label"><svg class="icon" aria-hidden="true"><use href="#i-scroll"/></svg>Số hiệu / phiên bản</div>
      <div class="meta-tile-value" title="${escapeHtml(doc.document_version)}">${escapeHtml(doc.document_version)}</div>
    </div>
    <div class="doc-meta-tile">
      <div class="meta-tile-label"><svg class="icon" aria-hidden="true"><use href="#i-calendar"/></svg>Ngày thu thập &amp; mã</div>
      <div class="meta-tile-value" title="${doc.id}">${doc.retrieved_at} • <code>${doc.id}</code></div>
    </div>
  `;

  // 2. Render Beautiful Formatted Markdown (Default View)
  const renderedContainer = document.getElementById('modal-doc-rendered');
  if (renderedContainer) {
    renderedContainer.innerHTML = renderFullMarkdown(doc.full_content);
  }

  // 3. Raw Markdown View
  elements.modalDocRaw.innerText = doc.full_content;

  // 4. Generate Chunks Breakdown with Chunk Cards
  const chunksGrid = document.getElementById('modal-doc-chunks');
  const chunkCountPill = document.getElementById('modal-chunk-count-pill');

  if (chunksGrid) {
    chunksGrid.innerHTML = `
      <div class="empty-state-hint">
        <div class="spinner"></div> Đang phân tích các mảnh ngữ nghĩa (chunks)...
      </div>
    `;

    fetch('/api/chunk-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: doc.full_content, strategy: 'heading', chunk_size: 700 }),
    })
      .then(r => r.json())
      .then(data => {
        const chunks = data.chunks || [];
        if (chunkCountPill) chunkCountPill.innerText = chunks.length;

        chunksGrid.innerHTML = chunks.map(c => `
          <div class="chunk-detail-card">
            <div class="chunk-detail-header">
              <div class="chunk-detail-id">
                <span class="chunk-rank">Chunk #${c.index}</span>
                <span class="chunk-len-badge">${c.length} ký tự (~${Math.round(c.length / 4)} tokens)</span>
              </div>
              <button class="chunk-btn-copy" onclick="copyChunkText(this)">
                <svg class="icon" aria-hidden="true"><use href="#i-copy"/></svg><span>Sao chép</span>
              </button>
            </div>
            <div class="chunk-detail-body">${escapeHtml(c.content)}</div>
          </div>
        `).join('');
      })
      .catch(err => {
        chunksGrid.innerHTML = `<div class="empty-state-hint">Lỗi tải chunks: ${err.message}</div>`;
      });
  }

  // 5. Default tab to "rendered"
  document.querySelectorAll('.doc-tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === 'rendered'));
  document.querySelectorAll('.doc-tab-content').forEach(c => c.classList.toggle('active', c.id === 'modal-tab-rendered'));

  // 6. Copy Full Content Button
  const btnCopyFull = document.getElementById('btn-copy-doc-content');
  if (btnCopyFull) {
    btnCopyFull.onclick = () => {
      navigator.clipboard.writeText(doc.full_content).then(() => {
        showToast('Đã sao chép toàn bộ nội dung tài liệu!');
      });
    };
  }

  elements.docModal.style.display = 'flex';
};

window.copyChunkText = function(btn) {
  const card = btn.closest('.chunk-detail-card') || btn.closest('.chunk-item-card');
  const textEl = card ? (card.querySelector('.chunk-detail-body') || card.querySelector('.chunk-body-text')) : null;
  const text = textEl ? textEl.innerText.trim() : '';
  if (text) {
    navigator.clipboard.writeText(text).then(() => {
      const origHtml = btn.innerHTML;
      btn.innerHTML = '<svg class="icon" aria-hidden="true"><use href="#i-check"/></svg><span>Đã chép</span>';
      setTimeout(() => { btn.innerHTML = origHtml; }, 1500);
      showToast('Đã sao chép nội dung chunk');
    });
  }
};

function escapeJsString(str) {
  return str
    .replace(/\\/g, '\\\\')
    .replace(/`/g, '\\`')
    .replace(/\$/g, '\\$');
}

// ================= TAB 3: PLAYGROUND =================
const SAMPLE_PRESETS = {
  utc_lib: `## 1. Thẻ thư viện của người học\nNgười học dùng thẻ đa năng — thẻ sinh viên, học viên, nghiên cứu sinh do các đơn vị trong trường ĐH GTVT cấp, có tích hợp chức năng sử dụng thư viện.\n\n## 2. Hạn mức mượn tài liệu về nhà của người học\nTài liệu được mượn về nhà bao gồm giáo trình và tài liệu tham khảo.\n- Giáo trình: không quá 10 cuốn / 1 lần mượn.\n- Tài liệu tham khảo: không quá 02 cuốn / 1 lần mượn.\nThời hạn mượn giáo trình là 1 học kỳ, tài liệu tham khảo là 14 ngày.\n\n## 3. Xử lý vi phạm và đền bù\nLàm mất tài liệu phải đền tài liệu mới hoặc bồi thường gấp 3 lần giá trị bìa sách.`,
  tmu_dorm: `## 1. Thời gian mở cửa, đóng cửa\n- Mở cửa: từ 5 giờ sáng.\n- Đóng cửa: từ 23 giờ đêm.\nCác trường hợp về muộn phải có lý do chính đáng và xuất trình thẻ nội trú cho ban quản lý.\n\n## 2. Quy định phòng ở nội trú\n- Sinh viên tự bảo quản tư trang cá nhân, không nấu ăn bằng bếp điện công suất lớn trong phòng.\n- Giữ gìn vệ sinh chung, tắt điện nước khi ra khỏi phòng.`,
  utc_policy: `## 1. Đối tượng miễn 100% học phí\n- Người có công với cách mạng và thân nhân theo Pháp lệnh ưu đãi người có công.\n- Sinh viên bị tàn tật, khuyết tật thuộc diện hộ nghèo hoặc hộ cận nghèo.\n- Sinh viên là người dân tộc thiểu số thuộc hộ nghèo và hộ cận nghèo.\n- Sinh viên dân tộc thiểu số rất ít người ở vùng có điều kiện kinh tế - xã hội khó khăn.\n\n## 2. Thủ tục và hồ sơ nộp\nHồ sơ bao gồm: Đơn xin miễn giảm học phí, Giấy chứng nhận hộ nghèo/cận nghèo có xác nhận của UBND xã/phường, Bản sao công chứng CCCD.`,
};

function initPlaygroundEvents() {
  // Preset change
  elements.presetChunkSample.addEventListener('change', (e) => {
    const text = SAMPLE_PRESETS[e.target.value] || '';
    elements.chunkInputText.value = text;
    updateChunkCharCounter();
  });
  // Trigger initial preset
  elements.chunkInputText.value = SAMPLE_PRESETS.utc_lib;
  updateChunkCharCounter();

  elements.chunkInputText.addEventListener('input', updateChunkCharCounter);

  // Run Chunk Comparison
  elements.btnRunChunkComparison.addEventListener('click', async () => {
    const text = elements.chunkInputText.value.trim();
    if (!text) {
      showToast('Vui lòng chọn hoặc nhập văn bản cần chia nhỏ');
      return;
    }

    const chunkSize = parseInt(elements.chunkParamSize.value, 10) || 700;
    const overlap = parseInt(elements.chunkParamOverlap.value, 10) || 0;

    elements.chunkingResultsGrid.innerHTML = `
      <div class="empty-state-hint" style="grid-column: 1 / -1;">
        <div class="spinner"></div> Đang phân tích so sánh 4 chiến lược...
      </div>
    `;

    const strategies = [
      { id: 'heading', name: 'HeadingChunker (Quy định L3A)', desc: 'Tách theo mục ## và giữ ngữ cảnh', recommended: true },
      { id: 'recursive', name: 'RecursiveChunker', desc: 'Đệ quy theo ký tự xuống dòng' },
      { id: 'fixed', name: 'FixedSizeChunker', desc: 'Cắt cứng theo số ký tự cố định' },
      { id: 'sentence', name: 'SentenceChunker', desc: 'Nhóm theo ranh giới dấu câu' },
    ];

    try {
      const results = await Promise.all(strategies.map(s =>
        fetch('/api/chunk-preview', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, strategy: s.id, chunk_size: chunkSize, overlap }),
        }).then(r => r.json())
      ));

      elements.chunkingResultsGrid.innerHTML = results.map((res, idx) => {
        const meta = strategies[idx];
        const previewSnippets = (res.chunks || []).slice(0, 3).map(c => `
          <div class="mini-chunk-snippet">#${c.index} (${c.length} chars): ${escapeHtml(c.content.slice(0, 80))}...</div>
        `).join('');

        return `
          <div class="strategy-stat-card ${meta.recommended ? 'best' : ''}">
            <div class="strat-card-header">
              <span class="strat-name">${meta.name}</span>
              ${meta.recommended ? '<span class="strat-badge recommended">Khuyên dùng</span>' : ''}
            </div>
            <div class="strat-metrics-row">
              <span>Chunks: <strong>${res.total_chunks}</strong></span>
              <span>Avg: <strong>${res.avg_len}</strong> ký tự</span>
              <span>Max: <strong>${res.max_len}</strong></span>
            </div>
            <div class="strat-chunks-preview">
              ${previewSnippets || '<em>Không có chunk</em>'}
            </div>
          </div>
        `;
      }).join('');
    } catch (err) {
      elements.chunkingResultsGrid.innerHTML = `<div class="empty-state-hint">Lỗi: ${err.message}</div>`;
    }
  });

  // Preset Similarity Pairs
  elements.presetSimilarityPairs.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'high') {
      elements.simText1.value = 'Bạn đọc có thẻ đa năng được mượn không quá 10 giáo trình và 02 tài liệu tham khảo.';
      elements.simText2.value = 'Sinh viên được phép mượn tối đa mười cuốn sách giáo trình về nhà để phục vụ học tập.';
    } else if (val === 'low') {
      elements.simText1.value = 'Bạn đọc có thẻ đa năng được mượn không quá 10 giáo trình thư viện.';
      elements.simText2.value = 'Sinh viên thuộc diện hộ nghèo được xét miễn giảm 100% học phí kỳ 1.';
    } else {
      elements.simText1.value = 'Quy chế đào tạo và đăng ký tín chỉ năm học 2025-2026.';
      elements.simText2.value = 'Thuật toán sắp xếp nhanh QuickSort triển khai bằng ngôn ngữ Python.';
    }
    elements.btnCalcSimilarity.click();
  });

  // Calculate Similarity
  elements.btnCalcSimilarity.addEventListener('click', async () => {
    const text1 = elements.simText1.value.trim();
    const text2 = elements.simText2.value.trim();
    if (!text1 || !text2) return;

    try {
      const res = await fetch('/api/similarity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text1, text2 }),
      });
      const data = await res.json();
      
      const score = data.cosine_similarity;
      elements.simScoreVal.innerText = score.toFixed(4);
      
      // Calculate meter width: 0.0 to 1.0 -> 0-100%
      const percentage = Math.max(0, Math.min(100, Math.round(score * 100)));
      elements.simMeterFill.style.width = percentage + '%';

      elements.simMeterFill.classList.remove('is-high', 'is-mid', 'is-low');
      if (score >= 0.70) {
        elements.simMeterFill.classList.add('is-high');
      } else if (score >= 0.35) {
        elements.simMeterFill.classList.add('is-mid');
      } else {
        elements.simMeterFill.classList.add('is-low');
      }
      
      elements.simExplanationText.innerHTML = `
        <strong>Phân tích:</strong> ${escapeHtml(data.explanation)}<br>
        <span style="font-size: 0.8rem; color: var(--text-muted);">
          Từ khóa chung: <code>${(data.common_keywords || []).join(', ') || 'không có'}</code>
        </span>
      `;
    } catch (err) {
      showToast('Lỗi tính cosine: ' + err.message);
    }
  });

  // Debounced input on text areas for real-time similarity feedback
  let simDebounceTimer = null;
  const handleSimInput = () => {
    clearTimeout(simDebounceTimer);
    simDebounceTimer = setTimeout(() => {
      elements.btnCalcSimilarity.click();
    }, 350);
  };
  elements.simText1.addEventListener('input', handleSimInput);
  elements.simText2.addEventListener('input', handleSimInput);

  // Run initial similarity calc & initial chunk comparison
  elements.btnCalcSimilarity.click();
  elements.btnRunChunkComparison.click();
}

function updateChunkCharCounter() {
  const len = elements.chunkInputText.value.length;
  elements.chunkCharCounter.innerText = `${len.toLocaleString()} ký tự`;
}

// ================= TAB 4: BENCHMARK =================
function initBenchmarkEvents() {
  elements.btnRunBenchmark.addEventListener('click', async () => {
    const strategy = elements.benchSelectStrategy.value;
    
    elements.btnRunBenchmark.disabled = true;
    elements.btnRunBenchmark.innerHTML = '<span class="spinner"></span><span>Đang chạy 5 queries...</span>';
    // Lượt chạy này nạp lại vector store cho cả 5 chiến lược nên mất khoảng
    // 20-30 giây. Không có đồng hồ đếm thì spinner trông như bị treo.
    elements.benchQueriesList.innerHTML = `
      <div class="empty-state-bench">
        <div class="spinner"></div>
        <p><strong>Đang chạy benchmark 5 câu hỏi trên 5 chiến lược chunking.</strong></p>
        <p>Mỗi chiến lược phải nhúng lại toàn bộ corpus, nên bước này thường mất 20-30 giây.</p>
        <p class="bench-elapsed">Đã chạy <strong id="bench-elapsed-val">0</strong> giây...</p>
      </div>
    `;
    const benchStartedAt = Date.now();
    const benchTicker = setInterval(() => {
      const el = document.getElementById('bench-elapsed-val');
      if (el) el.innerText = Math.round((Date.now() - benchStartedAt) / 1000);
    }, 1000);

    try {
      const res = await fetch(`/api/benchmark?strategy=${encodeURIComponent(strategy)}`);
      if (!res.ok) throw new Error('Không thể chạy benchmark');
      const data = await res.json();

      // Update Hero Score Radial
      elements.benchTotalScore.innerText = `${data.total_score} / ${data.max_score}`;
      elements.benchPercentage.innerText = `Đạt ${data.percentage}% • ${data.total_score >= 8 ? 'Đạt chuẩn xuất sắc' : 'Cần tối ưu thêm'}`;

      // Render 5 Query Cards
      elements.benchQueriesList.innerHTML = (data.results || []).map(q => {
        const scoreClass = q.score === 2 ? 'score-pill-2' : (q.score === 1 ? 'score-pill-1' : 'score-pill-0');
        
        const retrievedHtml = (q.retrieved || []).map(r => `
          <div class="rank-box ${r.is_gold ? 'is-gold' : ''}">
            <div class="rank-box-top">
              <span>Top ${r.rank} ${r.is_gold ? 'GOLD' : ''}</span>
              <span class="score-badge">${r.score}</span>
            </div>
            <div style="font-weight: 600; color: var(--text-primary);">${escapeHtml(r.doc_id)}</div>
            <div style="color: var(--text-muted); font-size: 0.72rem;">Audience: ${r.audience}</div>
            <div style="margin-top: 4px; font-size: 0.75rem; color: var(--text-secondary);">${escapeHtml(r.snippet)}</div>
          </div>
        `).join('');

        return `
          <div class="bench-query-card">
            <div class="bench-query-header">
              <div class="bench-q-id-title">
                <span class="bench-q-badge">${q.id}</span>
                <div>
                  <div class="bench-q-title">${escapeHtml(q.question)}</div>
                  <span style="font-size: 0.78rem; color: var(--text-muted);">${escapeHtml(q.school)}</span>
                </div>
              </div>
              <div class="bench-score-pill ${scoreClass}">${q.score} / ${q.max_score} điểm</div>
            </div>

            <div class="bench-gold-box">
              <div><strong>Tài liệu chuẩn (Gold Doc):</strong> <code>${escapeHtml(q.gold_doc)}</code></div>
              <div><strong>Đáp án chuẩn:</strong> ${escapeHtml(q.gold_answer)}</div>
              <div>
                <strong>Kiểm tra nội dung (must_contain):</strong> 
                ${q.must_contain_present 
                  ? '<span class="verdict-ok"><svg class="icon" aria-hidden="true"><use href="#i-check"/></svg>Đã xuất hiện đầy đủ trong ngữ cảnh</span>' 
                  : '<span class="verdict-bad"><svg class="icon" aria-hidden="true"><use href="#i-x"/></svg>Ngữ cảnh chưa chứa đủ chuỗi đặc trưng</span>'}
              </div>
            </div>

            ${q.ab_result ? `
              <div class="ab-test-alert">
                <strong><svg class="icon" aria-hidden="true"><use href="#i-flask"/></svg>A/B Testing bộ lọc metadata:</strong><br>
                • Có filter (<code>audience: student</code>): Truy xuất chính xác quy định người học <code>utc-thu-vien-sinh-vien</code>.<br>
                • Nếu KHÔNG dùng filter: Điểm số có thể bị nhiễu do tài liệu cán bộ/giảng viên (<code>utc-thu-vien-can-bo</code>) cùng từ khóa lọt vào top!
              </div>
            ` : ''}

            <div>
              <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 0.45rem; color: var(--text-secondary);">Top 3 Chunks Được Truy Xuất:</div>
              <div class="bench-retrieved-ranks">
                ${retrievedHtml}
              </div>
            </div>
          </div>
        `;
      }).join('');

      // Render Phụ Lục A: So sánh toàn diện 5 chiến lược Chunking
      const compWrapper = document.getElementById('strategies-comparison-table-wrapper');
      if (compWrapper && data.strategies_comparison) {
        compWrapper.innerHTML = `
          <table class="strategies-comparison-table">
            <thead>
              <tr>
                <th>Chiến Lược Chunking</th>
                <th>Tham Số Cấu Hình</th>
                <th>Tổng Chunks</th>
                <th>Độ Dài TB</th>
                <th>Điểm Benchmark</th>
                <th>Tỷ Lệ</th>
                <th>Nhận Xét &amp; Đánh Giá Kỹ Thuật</th>
              </tr>
            </thead>
            <tbody>
              ${data.strategies_comparison.map(s => `
                <tr class="${s.is_current ? 'highlight-strategy' : ''}">
                  <td>
                    <strong>${escapeHtml(s.name)}</strong>
                    ${s.badge ? ` <span class="badge-winner">${escapeHtml(s.badge)}</span>` : ''}
                    ${s.is_current ? ` <span class="badge badge-purple" style="margin-left: 6px;">Đang chọn</span>` : ''}
                  </td>
                  <td><code>${escapeHtml(s.params)}</code></td>
                  <td><strong>${s.total_chunks}</strong> chunks</td>
                  <td>${s.avg_len} chars</td>
                  <td>
                    <span class="rank-trophy">${s.score_display}</span>
                  </td>
                  <td><strong>${s.percentage}%</strong></td>
                  <td style="color: var(--text-soft); font-size: 0.72rem;">${escapeHtml(s.notes)}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
      }

      showToast(`Benchmark hoàn tất: ${data.total_score}/${data.max_score} điểm!`);
    } catch (err) {
      elements.benchQueriesList.innerHTML = `<div class="empty-state-bench">Lỗi: ${err.message}</div>`;
    } finally {
      clearInterval(benchTicker);
      elements.btnRunBenchmark.disabled = false;
      elements.btnRunBenchmark.innerHTML = '<svg class="icon" aria-hidden="true"><use href="#i-play"/></svg><span>Chạy lại benchmark</span>';
    }
  });
}

// ================= MODALS & COMMON =================
function initModals() {
  // Guide Modal
  elements.guideBtn.addEventListener('click', () => {
    elements.guideModal.style.display = 'flex';
  });
  elements.closeGuideBtns.forEach(btn => {
    btn?.addEventListener('click', () => {
      elements.guideModal.style.display = 'none';
    });
  });

  // Doc Details Modal
  elements.btnCloseDocModal.addEventListener('click', () => elements.docModal.style.display = 'none');
  elements.btnCloseDocModalBottom.addEventListener('click', () => elements.docModal.style.display = 'none');

  // Document Modal Tabs Switcher (Rendered, Chunks, Raw)
  document.querySelectorAll('.doc-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.doc-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.tab;
      document.querySelectorAll('.doc-tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `modal-tab-${target}`);
      });
    });
  });

  // Prompt Modal
  elements.btnClosePromptModal.addEventListener('click', () => elements.promptModal.style.display = 'none');
  elements.btnClosePromptModalBottom.addEventListener('click', () => elements.promptModal.style.display = 'none');

  elements.btnCopyPrompt.addEventListener('click', () => {
    navigator.clipboard.writeText(elements.modalPromptContent.innerText).then(() => {
      showToast('Đã sao chép prompt vào clipboard!');
    });
  });

  // Close modals when clicking outside window
  window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-backdrop')) {
      e.target.style.display = 'none';
    }
  });
}

window.openPromptModal = function() {
  elements.modalPromptContent.innerText = state.currentPrompt || 'Không có prompt để hiển thị.';
  elements.promptModal.style.display = 'flex';
};

function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ================= BOOTSTRAP =================
// Đặt ở cuối file để mọi `const` phía trên (SAMPLE_PRESETS, state, elements…)
// đã được khởi tạo xong trước khi initApp() chạm tới chúng.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
