import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import heapq
import re
from urllib.parse import unquote, quote
from collections import deque, defaultdict
import concurrent.futures
import networkx as nx
import plotly.graph_objects as go
import random

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="WikiNav Elite | Precision Pathfinder",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Hacker/Dashboard" Aesthetic
st.markdown("""
<style>
    /* Dark Theme Adjustments */
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 12px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .log-container {
        font-family: 'Courier New', monospace;
        background-color: #000;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
        height: 200px;
        overflow-y: auto;
        font-size: 12px;
        border: 1px solid #333;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        color: white;
        font-weight: bold;
    }
    /* Spinner override */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CORE UTILITIES & TELEMETRY ---

class Telemetry:
    """Real-time performance tracking."""
    def __init__(self):
        self.start_time = time.time()
        self.pages_scanned = 0
        self.links_found = 0
        self.pruned_count = 0
        self.cache_hits = 0
        self.requests_sent = 0
        self.current_depth = 0
        self.active_direction = "Init"
        self.logs = []
        self.max_logs = 50

    def log(self, message, type="info"):
        timestamp = f"[{time.time() - self.start_time:.2f}s]"
        entry = f"{timestamp} {message}"
        self.logs.insert(0, entry)
        if len(self.logs) > self.max_logs:
            self.logs.pop()

    def get_metrics(self):
        duration = time.time() - self.start_time
        rate = self.pages_scanned / duration if duration > 0 else 0
        return {
            "duration": duration,
            "scanned": self.pages_scanned,
            "rate": rate,
            "pruned": self.pruned_count,
            "links": self.links_found,
            "requests": self.requests_sent
        }

# --- 3. HEURISTICS & INTELLIGENCE LAYER ---

class HeuristicEngine:
    """AI-lite logic for scoring links and pruning."""
    
    def __init__(self, target_title, precision_mode="Balanced"):
        # --- FIX: Define stop_words FIRST ---
        self.stop_words = {"the", "of", "and", "in", "to", "a", "is", "for", "on", "by", "with"}
        
        # Now we can safely call tokenize
        self.target_tokens = set(self._tokenize(target_title))
        self.target_title_clean = target_title.lower().replace("_", " ")
        self.precision_mode = precision_mode
        
        # Regex filters for "garbage" pages
        self.filter_patterns = [
            r"^Category:", r"^File:", r"^Template:", r"^Talk:", r"^User:", 
            r"^Wikipedia:", r"^Help:", r"^Portal:", r"^Special:", r"^List_of",
            r"^\d{4}$", r"^\d{4}_in_", r"^ISBN_", r"^Main_Page$"
        ]

    def _tokenize(self, text):
        """Simple tokenizer."""
        clean = text.lower().replace("_", " ")
        # This line was failing because self.stop_words wasn't defined yet
        return [w for w in clean.split() if w.isalnum() and w not in self.stop_words]

    def is_pruned(self, title):
        """Determine if a page should be skipped entirely."""
        # 1. Regex Filter
        for pattern in self.filter_patterns:
            if re.match(pattern, title, re.IGNORECASE):
                return True
        
        # 2. Heuristic Filter (Aggressive Mode)
        if self.precision_mode == "Fast":
            if "disambiguation" in title.lower(): return True
            
        return False

    def score_link(self, link_title, link_text):
        """
        Assign a priority score (lower is better for Priority Queue).
        We invert logic here: High relevance = Low Score (0 is best).
        """
        score = 50.0 # Base score (neutral)
        
        link_clean = link_title.lower().replace("_", " ")
        text_clean = link_text.lower()
        
        # 1. Exact Title Match (Jackpot)
        if link_clean == self.target_title_clean:
            return 0.0
        
        # 2. Token Overlap
        link_tokens = set(self._tokenize(link_clean))
        overlap = len(link_tokens.intersection(self.target_tokens))
        
        if overlap > 0:
            score -= (overlap * 10) # Heavy bonus for sharing words
            
        # 3. Contextual Bonus
        # If the link text itself contains target keywords
        text_tokens = set(self._tokenize(text_clean))
        text_overlap = len(text_tokens.intersection(self.target_tokens))
        score -= (text_overlap * 2)

        # 4. Semantic Penalties
        if "outline" in link_clean or "index" in link_clean:
            score += 20
        
        return max(0.1, score)

# --- 4. NETWORK LAYER ---

class WikiSession:
    """Manages HTTP connections with caching and rate limits."""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers = {'User-Agent': 'WikiNavElite/2.0 (Education/Research)'}
        self.api_url = "https://en.wikipedia.org/api/rest_v1/page/html/"
        self.cache = {} # Simple in-memory cache

    def fetch_links(self, title):
        """Fetches page and extracts all valid internal links."""
        if title in self.cache:
            return self.cache[title], True # Cached
        
        try:
            url = f"{self.api_url}{quote(title)}"
            response = self.session.get(url, timeout=3)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            # Extract links with their anchor text for heuristic scoring
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('./') and ':' not in href[2:]:
                    clean_ref = unquote(href[2:].split('#')[0])
                    link_text = a.get_text()
                    links.append((clean_ref, link_text))
            
            # Deduplicate preserving order
            unique_links = list(dict.fromkeys(links))
            self.cache[title] = unique_links
            return unique_links, False
            
        except Exception as e:
            return [], False

# --- 5. SEARCH ENGINE (BIDIRECTIONAL) ---

class PathFinder:
    def __init__(self, start, end, mode="Balanced", max_threads=4):
        self.start = start.replace(" ", "_")
        self.end = end.replace(" ", "_")
        self.session = WikiSession()
        self.telemetry = Telemetry()
        self.heuristics = HeuristicEngine(self.end, mode)
        self.max_threads = max_threads
        
        # Configuration based on mode
        modes = {
            "Fast": {"depth": 3, "beam": 15},
            "Balanced": {"depth": 4, "beam": 30},
            "Deep": {"depth": 6, "beam": 50}
        }
        self.config = modes.get(mode, modes["Balanced"])

    def search(self):
        """
        Generator that yields telemetry updates.
        Implements Bidirectional Best-First Search.
        """
        # Data Structures
        # Queue item: (Priority, Depth, Title)
        f_queue = [(0, 0, self.start)] 
        b_queue = [(0, 0, self.end)]
        
        # Parents: {child: parent}
        f_parents = {self.start: None}
        b_parents = {self.end: None}
        
        visited_f = {self.start}
        visited_b = {self.end}
        
        step_count = 0
        
        while f_queue and b_queue:
            step_count += 1
            
            # 1. Decide Direction (Always expand smaller frontier for efficiency)
            if len(f_queue) <= len(b_queue):
                direction = "Forward"
                active_q = f_queue
                active_visited = visited_f
                active_parents = f_parents
                opposing_visited = visited_b
                opposing_parents = b_parents
            else:
                direction = "Backward"
                active_q = b_queue
                active_visited = visited_b
                active_parents = b_parents
                opposing_visited = visited_f
                opposing_parents = f_parents

            self.telemetry.active_direction = direction
            
            # 2. Pop Best Candidate (Priority Queue)
            try:
                prio, depth, current = heapq.heappop(active_q)
            except IndexError:
                break # Queue exhausted
            
            # Depth check
            if depth >= self.config["depth"]:
                continue

            # 3. Fetch Links
            raw_links, is_cached = self.session.fetch_links(current)
            
            # Telemetry updates
            self.telemetry.requests_sent += 1 if not is_cached else 0
            self.telemetry.cache_hits += 1 if is_cached else 0
            self.telemetry.pages_scanned += 1
            self.telemetry.links_found += len(raw_links)
            
            # 4. Process & Filter Links
            valid_candidates = []
            
            for link_title, link_text in raw_links:
                link_title = link_title.replace("_", " ") # Normalize
                
                # Check Intersection (Success!)
                if link_title in opposing_visited:
                    self.telemetry.log(f"🔗 CONNECTION FOUND AT: {link_title}", "success")
                    path = self._reconstruct_path(current, link_title, f_parents, b_parents, direction)
                    yield "FOUND", path
                    return

                # Filter/Prune
                if link_title in active_visited:
                    continue
                
                if self.heuristics.is_pruned(link_title):
                    self.telemetry.pruned_count += 1
                    continue
                
                # Score
                score = self.heuristics.score_link(link_title, link_text)
                valid_candidates.append((score, link_title))

            # 5. Add to Queue (Beam Search - limit number of children added)
            # Sort by score, take top N
            valid_candidates.sort(key=lambda x: x[0])
            top_candidates = valid_candidates[:self.config["beam"]]
            
            for score, title in top_candidates:
                active_visited.add(title)
                active_parents[title] = current
                heapq.heappush(active_q, (score + depth, depth + 1, title)) # A* cost: g(n) + h(n)
            
            # Log specific event
            if step_count % 2 == 0: # Reduce log noise
                best_link = top_candidates[0][1] if top_candidates else "None"
                self.telemetry.log(f"Expanded: {current[:20]}... | Best: {best_link[:15]}... | Q: {len(active_q)}")

            # Yield control back to UI
            yield "UPDATE", self.telemetry

        yield "FAILED", None

    def _reconstruct_path(self, meet_node, connection_node, f_parents, b_parents, direction):
        """Stitches the two halves of the path together."""
        
        # Build forward half
        path_start = []
        curr = meet_node if direction == "Forward" else connection_node
        while curr:
            path_start.append(curr)
            curr = f_parents.get(curr)
        path_start.reverse()
        
        # Build backward half
        path_end = []
        curr = connection_node if direction == "Forward" else meet_node
        while curr:
            path_end.append(curr)
            curr = b_parents.get(curr)
        
        # Join (removing duplicate meeting point)
        if direction == "Forward":
            # path_start ends with meet_node
            # path_end starts with connection_node (which IS meet_node's child/neighbor logic in bidirectional is tricky)
            # Actually, in this logic: meet_node (current) -> connection_node (already in other set)
            return path_start + path_end
        else:
            return path_start + path_end

# --- 6. UI COMPONENT LAYOUT ---

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Control Center")
        
        mode = st.radio("Search Mode", ["Fast", "Balanced", "Deep"], index=1,
                        help="Fast: Depth 3, High Pruning. Deep: Depth 6, Low Pruning.")
        
        st.markdown("---")
        st.markdown("**Search Algorithm**")
        st.info("Bidirectional Best-First Search")
        st.markdown("**Heuristic Model**")
        st.info("TF-IDF Keyword Matching")
        
        st.markdown("---")
        st.caption("v2.4.0-Elite | Python 3.10+")

    # --- Main Header ---
    st.markdown("""
        <h1 style='text-align: center; color: white; margin-bottom: 0px;'>
            WIKI<span style='color: #3b82f6;'>NAV</span> ELITE
        </h1>
        <p style='text-align: center; color: #9ca3af; margin-bottom: 30px;'>
            High-Performance Heuristic Wikipedia Pathfinder
        </p>
    """, unsafe_allow_html=True)

    # --- Input Section ---
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        start_input = st.text_input("START POINT", "Python (programming language)")
    with c2:
        end_input = st.text_input("TARGET DESTINATION", "Artificial intelligence")
    with c3:
        st.write("") # Spacer
        st.write("") # Spacer
        run_btn = st.button("INITIATE SEARCH 🚀")

    # --- Visual Containers ---
    if 'searching' not in st.session_state:
        st.session_state.searching = False

    # Placeholders for Real-time Data
    metrics_ph = st.empty()
    logs_ph = st.empty()
    result_ph = st.empty()

    if run_btn:
        st.session_state.searching = True
        
        # Initialize Engine
        finder = PathFinder(start_input, end_input, mode)
        
        # Progress Bar
        progress_bar = st.progress(0)
        
        # Search Loop
        try:
            for status, data in finder.search():
                if status == "UPDATE":
                    telemetry = data
                    stats = telemetry.get_metrics()
                    
                    # Update Metrics Panel (Custom HTML for speed/look)
                    metrics_html = f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div class="metric-card" style="width: 24%;">
                            <div class="metric-value">{stats['scanned']}</div>
                            <div class="metric-label">Pages Scanned</div>
                        </div>
                        <div class="metric-card" style="width: 24%;">
                            <div class="metric-value">{stats['rate']:.1f}</div>
                            <div class="metric-label">Pages / Sec</div>
                        </div>
                        <div class="metric-card" style="width: 24%;">
                            <div class="metric-value">{stats['pruned']}</div>
                            <div class="metric-label">Pruned</div>
                        </div>
                        <div class="metric-card" style="width: 24%;">
                            <div class="metric-value" style="color: #10b981;">{telemetry.active_direction}</div>
                            <div class="metric-label">Active Frontier</div>
                        </div>
                    </div>
                    """
                    metrics_ph.markdown(metrics_html, unsafe_allow_html=True)
                    
                    # Update Logs
                    log_text = "<br>".join([f"<span style='color: #00ff00;'>></span> {l}" for l in telemetry.logs[:8]])
                    logs_ph.markdown(f"<div class='log-container'>{log_text}</div>", unsafe_allow_html=True)
                    
                    # Pulse Progress
                    progress_bar.progress(min(stats['scanned'] % 100, 100))

                elif status == "FOUND":
                    path = data
                    st.balloons()
                    progress_bar.progress(100)
                    
                    # Final Stats
                    total_time = finder.telemetry.get_metrics()['duration']
                    
                    # --- Result Rendering ---
                    result_html = f"""
                    <div style="background-color: #064e3b; padding: 20px; border-radius: 10px; border: 1px solid #059669; text-align: center; margin-top: 20px;">
                        <h2 style="color: #34d399; margin:0;">PATH ACQUIRED</h2>
                        <p style="color: #a7f3d0;">Execution Time: {total_time:.2f}s | Path Length: {len(path)-1} clicks</p>
                    </div>
                    """
                    result_ph.markdown(result_html, unsafe_allow_html=True)
                    
                    # Visual Path Chain
                    st.write("")
                    st.subheader("📍 Trajectory")
                    
                    # Generate Network Graph
                    G = nx.DiGraph()
                    for i in range(len(path)-1):
                        G.add_edge(path[i], path[i+1])
                        st.markdown(f"""
                        <div style="padding: 10px; border-left: 3px solid #3b82f6; background: #1f2937; margin-bottom: 5px;">
                            <span style="color: #9ca3af; font-size: 12px;">Step {i}</span><br>
                            <a href="https://en.wikipedia.org/wiki/{path[i].replace(' ', '_')}" target="_blank" style="color: white; font-weight: bold; text-decoration: none; font-size: 18px;">
                                {path[i]}
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Final Target
                    st.markdown(f"""
                        <div style="padding: 10px; border-left: 3px solid #10b981; background: #1f2937;">
                            <span style="color: #9ca3af; font-size: 12px;">TARGET</span><br>
                            <a href="https://en.wikipedia.org/wiki/{path[-1].replace(' ', '_')}" target="_blank" style="color: #34d399; font-weight: bold; text-decoration: none; font-size: 18px;">
                                {path[-1]}
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    break

                elif status == "FAILED":
                    result_ph.error("Search Exhausted: No path found within parameters.")
                    break
                    
        except Exception as e:
            st.error(f"Critical System Error: {e}")

if __name__ == "__main__":
    main()
