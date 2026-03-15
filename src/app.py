import streamlit as st
import streamlit.components.v1 as components
import eda
import prediction
import os

# SET PAGE CONFIG (HARUS DI PALING ATAS)
st.set_page_config(page_title="DemandSense AI", layout="wide")

# CSS & JS INJECTORS
def local_css(file_name):
    """Fungsi untuk memuat file style.css eksternal."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def inject_anime_js():
    """Menyuntikkan library Anime.js dan script animasi ke elemen selectbox."""
    anime_script = """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
    <script>
        // Gunakan MutationObserver agar tetap jalan meskipun Streamlit rerun
        const observer = new MutationObserver((mutations) => {
            const target = document.querySelector('.stSelectbox');
            if (target && !target.dataset.animated) {
                target.dataset.animated = "true"; // Penanda agar tidak diduplikasi
                
                // Efek Scale pas Mouse Masuk (Hover)
                target.addEventListener('mouseenter', () => {
                    anime({
                        targets: '.stSelectbox',
                        scale: 1.02,
                        duration: 400,
                        easing: 'easeOutExpo'
                    });
                });
                
                // Balikin Scale pas Mouse Keluar
                target.addEventListener('mouseleave', () => {
                    anime({
                        targets: '.stSelectbox',
                        scale: 1,
                        duration: 400,
                        easing: 'easeOutExpo'
                    });
                });
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """
    components.html(anime_script, height=0)

def main():
    # Load assets
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_css(os.path.join(current_dir, "style.css"))
    inject_anime_js()

    # SIDEBAR WITH BOOTSTRAP DESIGN 
    # Load Bootstrap CSS
    st.sidebar.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">', unsafe_allow_html=True)
    
    with st.sidebar:
        # Header Animated
        st.markdown("<h1 class='animated-sidebar-text'>DemandSense</h1>", unsafe_allow_html=True)
        st.markdown("---")


        # Practitioner Card (Bootstrap & Glassmorphism)
        st.markdown("""
            <div class="card bg-dark border-secondary mb-4 shadow-sm" style="background: rgba(255,255,255,0.05) !important; backdrop-filter: blur(10px);">
                <div class="card-body p-3">
                    <div class="d-flex align-items-center mb-2">
                        <div class="badge bg-primary me-2">Time Series</div>
                        <span class="fw-bold text-white small">Practitioner Info</span>
                    </div>
                    <p class="text-light small mb-1"><strong>Name:</strong> Risyadhana Syaifuddin</p>
                    <p class="text-light small mb-0"><strong>Project ID:</strong> DemandSense AI</p>
                </div>
            </div>
            <p class='text-white fw-bold mb-1 small' style='opacity: 0.8;'>Navigate Menu:</p>
        """, unsafe_allow_html=True)
        
        # Selectbox (Target Anime.js Animation)
        menu = st.selectbox(
            "Navigate Menu:", 
            ["EDA Analysis", "Demand Forecasting"], 
            label_visibility="collapsed"
        )
        st.markdown("---")
        
    # NAVIGATION LOGIC
    if menu == "EDA Analysis":
        eda.run()
    else:
        prediction.run()

if __name__ == "__main__":
    main()