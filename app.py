import streamlit as st
import requests
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import unquote
import time

# --- Configuration ---
st.set_page_config(
    page_title="Wiki Speedrun Pathfinder",
    page_icon="🔗",
    layout="centered"
)

# --- Logic Class (Adapted from your script) ---
class WikipediaPathfinder:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/wiki/"
        self.api_url = "https://en.wikipedia.org/api/rest_v1/page/html/"
        self.visited = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikipediaPathfinder/1.0'
        })
    
    def get_links(self, article_title):
        """Extract all valid Wikipedia article links from a page."""
        try:
            # Check for empty title
            if not article_title:
                return set()
                
            url = self.api_url + requests.utils.quote(article_title)
            response = self.session.get(url, timeout=5) # Reduced timeout for UI responsiveness
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Only process internal Wikipedia article links
                if href.startswith('./'):
                    article_name = href[2:]  # Remove './'
                    # Remove anchors and decode
                    article_name = unquote(article_name.split('#')[0]).replace('_', ' ')
                    
                    # Filter out special pages, files, categories, etc.
                    if ':' not in article_name and article_name:
                        links.add(article_name)
            
            return links
        
        except Exception as e:
            # We silently fail on individual link fetch errors to keep the search going
            return set()
    
    def normalize_title(self, title):
        """Normalize Wikipedia article title."""
        return title.strip().replace('_', ' ')
    
    def find_path(self, start_article, target_article, max_depth, progress_callback):
        """
        Find the shortest path from start to target using BFS.
        Updates UI via progress_callback.
        """
        start_article = self.normalize_title(start_article)
        target_article = self.normalize_title(target_article)
        
        start_time = time.time()
        
        # BFS queue: (current_article, path_to_current)
        queue = deque([(start_article, [start_article])])
        self.visited = {start_article.lower()}
        
        pages_visited = 0
        
        while queue:
            current_article, path = queue.popleft()
            pages_visited += 1
            
            # Update UI every 5 pages to avoid slowing down execution too much
            if pages_visited % 5 == 0:
                progress_callback(pages_visited, current_article, len(path)-1)
            
            # Check depth
            if len(path) > max_depth + 1: # +1 because path includes start node
                continue
            
            # Check if we reached the target
            if current_article.lower() == target_article.lower():
                elapsed_time = time.time() - start_time
                return {
                    "success": True,
                    "path": path,
                    "pages_visited": pages_visited,
                    "time": elapsed_time
                }
            
            # Get all links from current article
            links = self.get_links(current_article)
            
            # Add unvisited links to queue
            for link in links:
                if link.lower() not in self.visited:
                    self.visited.add(link.lower())
                    queue.append((link, path + [link]))
        
        # No path found
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "pages_visited": pages_visited,
            "time": elapsed_time
        }

# --- Streamlit UI ---

st.title("🔗 Wikipedia Speedrun Pathfinder")
st.markdown("""
Find the shortest path between two Wikipedia articles using only internal links.
""")

col1, col2 = st.columns(2)
with col1:
    start_input = st.text_input("Starting Article", value="Potato")
with col2:
    end_input = st.text_input("Target Article", value="Barack Obama")

with st.expander("⚙️ Advanced Settings"):
    max_depth = st.slider("Max Search Depth (Clicks)", min_value=1, max_value=6, value=3, 
                          help="Higher depth takes exponentially longer.")
    st.caption("Note: Depth 3 is usually sufficient. Going above 4 may result in timeouts.")

if st.button("🚀 Find Path", type="primary"):
    if not start_input or not end_input:
        st.error("Please enter both a start and target article.")
    else:
        # Initialize Pathfinder
        pathfinder = WikipediaPathfinder()
        
        # Create a placeholder for live status updates
        status_container = st.status("Searching Wikipedia...", expanded=True)
        progress_text = status_container.empty()
        log_container = status_container.empty()
        
        def update_progress(count, current, depth):
            progress_text.markdown(f"**Pages Scanned:** `{count}`")
            log_container.text(f"Exploring: {current} (Depth: {depth})")
        
        # Run Search
        result = pathfinder.find_path(start_input, end_input, max_depth, update_progress)
        
        status_container.update(label="Search Complete!", state="complete", expanded=False)
        
        if result["success"]:
            st.success("✅ Path Found!")
            
            # Display Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Time Taken", f"{result['time']:.2f}s")
            m2.metric("Pages Scanned", result['pages_visited'])
            m3.metric("Clicks Required", len(result['path']) - 1)
            
            st.divider()
            
            # visual Path
            st.subheader("📍 The Route")
            
            path = result['path']
            for i, article in enumerate(path):
                if i == 0:
                    st.markdown(f"🟢 **START**: [{article}](https://en.wikipedia.org/wiki/{article.replace(' ', '_')})")
                elif i == len(path) - 1:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⬇️")
                    st.markdown(f"🔴 **TARGET**: [{article}](https://en.wikipedia.org/wiki/{article.replace(' ', '_')})")
                else:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⬇️")
                    st.markdown(f"⚪ *Step {i}*: [{article}](https://en.wikipedia.org/wiki/{article.replace(' ', '_')})")
            
            st.divider()
            st.code(" → ".join(path), language=None)
            
        else:
            st.error(f"❌ No path found within {max_depth} clicks.")
            st.info(f"Scanned {result['pages_visited']} pages in {result['time']:.2f} seconds.")
            st.markdown("Try increasing the depth or choosing more related topics.")
