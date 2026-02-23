tell application "Keynote"
    activate
    -- Create a new document with the default theme
    set newDoc to make new document with properties {document theme:theme "White"}
    
    tell newDoc
        -- --- Slide 1: Title Slide ---
        set slide1 to make new slide with properties {base layout:layout "Title & Subtitle"}
        tell slide1
            set object text of default title item to "Hustle & Have Fun: Redefining UX/PM/Eng Collaboration"
            set object text of default body item to "Building the Software Stack 'Community Engine' in 3 Months\n\nPresenters: UX, Brittany (PM), Rob (Eng)"
            set presenter notes of it to "Hi everyone. Today, Brittany, Rob, and I want to share a story about how we completely reimagined how UX, Product, and Engineering collaborate. By leaning into the principles of 'Stay Scrappy' and 'Hustle & Have Fun', we moved from a blank whiteboard to a fully functional 'Community Engine' MVP in exactly 3 months."
        end tell
        
        -- --- Slide 2: The Opportunity & The "Scrappy" Shift ---
        set slide2 to make new slide with properties {base layout:layout "Title & Bullets"}
        tell slide2
            set object text of default title item to "The Opportunity & The 'Scrappy' Shift"
            set object text of default body item to "• The Problem: 'Black Box' development and Reviewer Bottlenecks were slowing down external TPU support contributions.
• The Opportunity: Shift from closed development to 'Open Acceleration'—enabling the community to drive performance enhancements (like vLLM) so internal engineers don't have to.
• The Shift in Mindset ('Stay Scrappy'): Stop standard design cycles. We brainstormed the 'Six-Stage Flywheel' and immediately built the automated foundation first, rather than pitching perfect mockups."
            set presenter notes of it to "The core issue was that community developers wanted to build on our TPU software stack, but faced a 'Black Box'. Internally, our engineers were drowning in bespoke requests and review bottlenecks. Instead of doing what we normally do—spending weeks on high-fidelity Figma mockups and PM requirements docs—we got scrappy. We defined a 'Six-Stage Flywheel' for community engagement, and immediately started writing code to build the foundational automation MVP."
        end tell
        
        -- --- Slide 3: The MVP – The "Community Engine" ---
        set slide3 to make new slide with properties {base layout:layout "Title & Bullets"}
        tell slide3
            set object text of default title item to "The MVP – The 'Community Engine'"
            set object text of default body item to "• What we built: A 3-step automated workflow requiring zero human intervention:
  1. Auto CUJ (The Truth Pipeline): Automated sweeps find TPU test failures and expose them in a transparent matrix.
  2. Auto Mission Board (Candy Crush): Those failures automatically generate 'Good First Issues' to lower the entry barrier for contributors.
  3. Auto Leaderboard: Triggers an automated 'Inclusive Contributor Wall' update when code is merged, rewarding the dopamine loop."
            set presenter notes of it to "Our MVP focused purely on the automated workflow to eliminate human overhead. We call it the Community Engine. Step 1 is the Auto CUJ, which automatically detects missing features and surfaces them. Step 2 is the 'Candy Crush' Mission Board: those gaps automatically turn into bite-sized GitHub issues. Step 3 closes the loop—when the community merges a fix, an automated Leaderboard updates to celebrate them. We went from concept to a live, automated ecosystem in 3 months."
        end tell
        
        -- --- Slide 4: Roles Reimagined ("Mission First") ---
        set slide4 to make new slide with properties {base layout:layout "Title & Bullets"}
        tell slide4
            set object text of default title item to "Roles Reimagined ('Mission First')"
            set object text of default body item to "• UX / Design: Erased the 'mockup' boundary. Wrote the actual working Python code and GitHub Actions to automate the MVP.
• Engineering (Rob): Shifted to an accelerator. Helped modularize the kernel tuner, solved complex infrastructure blocks, and reviewed the designer's code.
• Product (Brittany): Managed the 'Hockey Stick'. Aligned the internal strategy, prioritized the 'Golden Path', and provided continuous input without stalling execution."
            set presenter notes of it to "The secret to doing this in 3 months was erasing our traditional job titles and working 'Mission First'. I stepped outside UX and actually wrote the working Python scripts and GitHub Actions to bring the Mission Board to life. Rob shifted from strictly writing features to acting as an accelerator—reviewing my code and modularizing the internal tuner. And Brittany focused on unblocking us and aligning our work with the wider TPU strategy without slowing down our daily execution."
        end tell
        
        -- --- Slide 5: The Impact & The Future ---
        set slide5 to make new slide with properties {base layout:layout "Title & Bullets"}
        tell slide5
            set object text of default title item to "The Impact & The Future"
            set object text of default body item to "• Impact: A live, scalable Contribution Ecosystem that reduces internal engineering load and increases external velocity.
• Key Takeaway: Proving value via working code (Stay Scrappy) builds momentum faster than pitching slides.
• Our Ask: How can leadership help us scale this embedded, high-velocity ('Hustle & Have Fun') team model to other critical infrastructure tracks?"
            set presenter notes of it to "The impact here isn't just a cool GitHub script. We've built a scalable engine that actively reduces our internal engineering load. What we learned is that proving value via working code builds unstoppable momentum. Our ask for you today is this: How can we scale this exact model—this high-velocity, boundary-less collaboration—to other critical infrastructure teams at Google?"
        end tell
        
        -- Clean up default empty slide
        delete slide 6
    end tell
end tell
