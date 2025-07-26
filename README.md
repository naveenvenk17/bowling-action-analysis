# 🏏 Cricket Bowling Consistency Analyzer

**Help bowlers get 1% better every day by identifying inconsistent joints at the release point**

Ever wondered why your bowling action feels inconsistent? Or why some deliveries are perfect while others go wayward? This AI-powered system analyzes 10 videos of your bowling action to identify which joints (elbow, knee, etc.) are wobbly or inconsistent at the critical moment of release - issues that are often impossible to spot with the naked eye.

## 🎯 The Problem We Solve

Bowling consistency is the difference between good and great bowlers. The problem? Inconsistent joints at the release point are nearly impossible to detect during practice or match play. Even the most experienced coaches can miss subtle biomechanical inconsistencies that cause:

- **Erratic line and length**
- **Inconsistent pace**
- **Increased injury risk**
- **Frustrating performance plateaus**

This system uses advanced computer vision to catch what your eyes can't see.

## 📸 See It In Action

![Cricket Analysis Interface](screenshots/screenshot1.png)
*Real-time pose estimation with joint angle measurements and AI-powered release point detection*

![Bowling Action Analysis](screenshots/screenshot2.png) 
*Frame-by-frame analysis showing complete biomechanical breakdown of bowling action*

## ✨ What Does It Do?

Think of this as your personal biomechanical coach that never gets tired. Upload 10 videos of your bowling action, and it will:

- **🎯 Automatically detect** the exact moment the ball is released in each video
- **📐 Calculate joint angles** throughout the bowling action for every delivery
- **🔍 Identify inconsistencies** in specific joints (elbow, knee, shoulder, etc.) at the release point
- **📊 Compare all 10 deliveries** to find patterns and inconsistencies
- **📈 Generate actionable insights** to help you improve by 1% every day
- **📋 Provide specific recommendations** for which joints to focus on

Perfect for bowlers, coaches, and anyone serious about improving their cricket performance through data-driven analysis.

## 🚀 Quick Start

Getting started is easy! Just run one command:

```bash
# Launch the web interface - upload your 10 videos and get insights!
python main.py --mode ui
```

Or if you prefer the command line:

```bash
# Analyze your bowling videos for consistency
python main.py --mode backend --videos delivery1.mp4 delivery2.mp4 delivery3.mp4 delivery4.mp4 delivery5.mp4 delivery6.mp4 delivery7.mp4 delivery8.mp4 delivery9.mp4 delivery10.mp4
```

That's it! The system will process your videos and tell you exactly which joints are inconsistent at the release point.

## 📦 Installation

First, make sure you have Python installed, then:

```bash
# Install everything you need
pip install -r requirements.txt

# You're ready to go!
python main.py --help
```

## 🎮 How to Use

### Method 1: Web Interface (Recommended for bowlers)
```bash
python main.py --mode ui
```

**What you'll see:**
- **Video Upload**: Upload 10 videos of your bowling action
- **AI Analysis**: Automatic release point detection with confidence scores
- **Consistency Report**: Which joints are wobbly or inconsistent
- **Interactive Display**: Click through frames to see the complete bowling action
- **Real-time Analysis**: Joint angles and pose estimation overlays
- **Actionable Insights**: Specific recommendations for improvement

![Interface Features](screenshots/screenshot1.png)
*Key features: AI suggestion (Frame 150, 85% confidence), pose tracking, and consistency analysis*

### Method 2: Command Line (Great for coaches analyzing multiple players)
```bash
# Analyze a player's bowling consistency
python main.py --mode backend \
  --videos delivery1.mp4 delivery2.mp4 delivery3.mp4 delivery4.mp4 delivery5.mp4 delivery6.mp4 delivery7.mp4 delivery8.mp4 delivery9.mp4 delivery10.mp4 \
  --export-all
```

## 🧠 The AI Behind It

The system combines several smart technologies to catch what you can't see:

- **Sports2D**: Tracks body movements and calculates joint angles with precision
- **YOLOv8**: Spots the ball and bowler in each frame
- **Smart algorithms**: Determine exactly when the ball leaves the hand
- **Statistical analysis**: Compares all 10 deliveries to find inconsistencies
- **Biomechanical analysis**: Identifies which joints are inconsistent at the release point

All of this happens automatically - you just provide the videos!

## 📋 What You Get

### Consistency Analysis
- **Joint-by-joint breakdown**: "Your right elbow shows 15° variation at release"
- **Inconsistency scoring**: "Knee angle varies by 8° across deliveries"
- **Release point stability**: "Release timing varies by ±3 frames"
- **Confidence scores**: "AI is 87% confident in this analysis"

### Visual Insights
The interface shows you exactly what's happening:
- **Pose skeleton**: Green lines showing body structure
- **Joint angles**: Precise measurements at each joint
- **Release detection**: AI identifies the optimal release frame
- **Frame navigation**: Step through the entire bowling action
- **Inconsistency highlighting**: Red markers show wobbly joints

### Detailed Reports
```csv
Video,Release Frame,Right Elbow Angle,Left Knee Angle,Hip Rotation,Consistency Score
delivery1.mp4,145,156.7°,142.3°,23.1°,85%
delivery2.mp4,148,158.2°,141.9°,22.7°,82%
delivery3.mp4,143,154.1°,143.1°,24.2°,78%
...
```

### Actionable Recommendations
- **"Focus on your right elbow - it's varying by 12° at release"**
- **"Your knee angle is consistent - good work there"**
- **"Release timing is stable - this is a strength"**
- **"Shoulder rotation needs work - varies by 8°"**

## ⚙️ Customization

Want to fine-tune the analysis? Edit the `config.toml` file:

```toml
# Quick consistency check
[profiles.fast]
sports2d.mode = "fast"
save_images = false

# Detailed biomechanical analysis  
[profiles.research]
sports2d.mode = "accurate"
yolo.confidence_threshold = 0.15
save_everything = true
```

## 🔧 For Developers

The system is built with a clean, modular architecture that's easy to extend:

```
🏗️ Project Structure
├── 📁 core/                    # Main analysis engine
├── 📁 utils/                   # Specialized components  
│   ├── video_analyzer.py    # Sports2D integration
│   ├── release_point_detector.py  # AI detection magic
│   ├── consistency_analyzer.py    # Joint consistency analysis
│   └── result_generator.py      # Data export wizardry
└── 📁 ui/                      # Streamlit web interface
```

### Want to integrate with your own app?

```python
from core.analyzer import CricketAnalyzer

# It's this simple
analyzer = CricketAnalyzer()
result = analyzer.analyze_consistency(["delivery1.mp4", "delivery2.mp4", ...])

print(f"Most inconsistent joint: {result.most_inconsistent_joint}")
print(f"Consistency score: {result.overall_consistency_score}")
```

## 🎯 Use Cases

**For Bowlers:**
- Identify which joints are inconsistent at release
- Get specific areas to focus on in practice
- Track improvement over time
- Achieve that 1% daily improvement

**For Coaches:**
- Compare your players' consistency
- Identify specific technical issues
- Track improvement over time
- Provide data-driven coaching

**For Researchers:**
- Biomechanical consistency studies
- Performance correlation research
- Injury prevention analysis

**For Cricket Academies:**
- Standardized consistency assessment
- Player development tracking
- Performance benchmarking

## 🎥 Live Demo

Here's what you'll see when analyzing your bowling consistency:

1. **Upload 10 videos** → System extracts frames automatically
2. **AI analyzes each frame** → Pose detection and joint angle calculation
3. **Consistency analysis** → "Right elbow varies by 12° at release"
4. **Interactive exploration** → Click through frames to see the complete action
5. **Get actionable insights** → "Focus on stabilizing your elbow angle"

## 🤝 Contributing

Found a bug? Have an idea? The modular design makes it easy to contribute:

1. **Add new consistency metrics** → Create modules in `utils/`
2. **Build new interfaces** → Add them in `ui/`
3. **Improve AI detection** → Enhance the detector algorithms
4. **Add export formats** → Extend the result generator

## 📄 What's Under the Hood

This system brings together several advanced technologies:

- **Computer Vision**: YOLOv8 for object detection
- **Pose Estimation**: Sports2D for biomechanical analysis  
- **Machine Learning**: AI algorithms for release point detection
- **Statistical Analysis**: Consistency measurement and comparison tools
- **Web Technologies**: Modern, responsive interface

But you don't need to understand any of that to use it - it just works!

## 🎉 Ready to Get 1% Better?

Whether you're a bowler looking to improve your consistency, a coach helping players develop, or a researcher studying biomechanics, this system gives you the tools to identify and fix the subtle issues that hold back performance.

```bash
# Let's get started on your 1% improvement journey!
python main.py --mode ui
```

---

*Built with ❤️ to help bowlers get 1% better every day* 