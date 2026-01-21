<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cloudera AI Demo Hub - Cataloging System</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script src="https://kit.fontawesome.com/your-fontawesome-kit.js" crossorigin="anonymous"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
            color: #2d3748;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #ed8936;
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        h1 {
            color: #dd6b20;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #4a5568;
            font-size: 1.2rem;
            margin-bottom: 20px;
        }
        
        .cloudera-badge {
            background: #ed8936;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
        }
        
        .content-wrapper {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        @media (max-width: 1024px) {
            .content-wrapper {
                grid-template-columns: 1fr;
            }
        }
        
        .flowchart-container {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 5px solid #ed8936;
        }
        
        #flowchart {
            min-height: 500px;
            background: #f8fafc;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .catalog-sidebar {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 5px solid #4299e1;
        }
        
        .section-title {
            color: #2d3748;
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ed8936;
            display: flex;
            align-items: center;
        }
        
        .section-title i {
            margin-right: 10px;
            color: #ed8936;
        }
        
        .current-status {
            margin-bottom: 30px;
        }
        
        .status-item {
            background: #f8fafc;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 4px solid #4299e1;
        }
        
        .status-label {
            font-weight: 600;
            color: #4a5568;
        }
        
        .status-value {
            color: #2d3748;
            font-weight: bold;
        }
        
        .status-complete {
            color: #38a169;
        }
        
        .status-current {
            color: #ed8936;
        }
        
        .status-pending {
            color: #dd6b20;
        }
        
        .catalog-preview {
            background: #f8fafc;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid #e2e8f0;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            overflow-x: auto;
        }
        
        .yaml-key { color: #2b6cb0; }
        .yaml-string { color: #38a169; }
        .yaml-number { color: #dd6b20; }
        .yaml-comment { color: #a0aec0; }
        
        .folder-structure {
            background: #f8fafc;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid #e2e8f0;
        }
        
        .folder {
            color: #2b6cb0;
            font-weight: 600;
        }
        
        .file {
            color: #4a5568;
        }
        
        .indent-1 { margin-left: 20px; }
        .indent-2 { margin-left: 40px; }
        .indent-3 { margin-left: 60px; }
        
        .tag-system {
            background: #fffaf0;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #ed8936;
        }
        
        .tag-category {
            display: inline-block;
            background: #feebc8;
            color: #9c4221;
            padding: 5px 10px;
            border-radius: 4px;
            margin: 5px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .phase-details {
            margin-top: 30px;
        }
        
        .phase {
            margin-bottom: 25px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .phase-1 {
            border-left-color: #38a169;
        }
        
        .phase-2 {
            border-left-color: #ed8936;
        }
        
        .phase-3 {
            border-left-color: #4299e1;
        }
        
        .phase-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .phase-title {
            font-size: 1.3rem;
            font-weight: bold;
            color: #2d3748;
        }
        
        .phase-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
        }
        
        .phase-steps {
            list-style-type: none;
        }
        
        .phase-step {
            padding: 10px 0;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: flex-start;
        }
        
        .phase-step:last-child {
            border-bottom: none;
        }
        
        .step-checkbox {
            margin-right: 10px;
            color: #38a169;
            font-size: 1.2rem;
        }
        
        .step-pending {
            color: #cbd5e0;
            font-size: 1.2rem;
            margin-right: 10px;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #718096;
            font-size: 0.9rem;
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        .highlight {
            background: #feebc8;
            padding: 3px 6px;
            border-radius: 4px;
            color: #9c4221;
            font-weight: 600;
        }
        
        .interactive-btn {
            background: #ed8936;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            margin: 10px 5px;
            transition: background 0.3s;
        }
        
        .interactive-btn:hover {
            background: #dd6b20;
        }
        
        .btn-group {
            display: flex;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="cloudera-badge">CLOUDERA AI DEMO HUB</div>
            <h1><i class="fas fa-boxes"></i> Cataloging System Flow</h1>
            <p class="subtitle">Smart catalog system for organizing AI/ML demos by business scenarios</p>
            <p>Focus: Multi-dimensional tagging system for discoverability across teams</p>
        </header>
        
        <div class="content-wrapper">
            <div class="flowchart-container">
                <h2 class="section-title"><i class="fas fa-project-diagram"></i> Cataloging System Flowchart</h2>
                <div id="flowchart"></div>
                
                <div class="btn-group">
                    <button class="interactive-btn" onclick="showPhase('phase1')">
                        <i class="fas fa-layer-group"></i> Show Phase 1 Details
                    </button>
                    <button class="interactive-btn" onclick="showPhase('phase2')">
                        <i class="fas fa-tags"></i> Show Cataloging System
                    </button>
                    <button class="interactive-btn" onclick="showPhase('phase3')">
                        <i class="fas fa-rocket"></i> Show Launch Process
                    </button>
                </div>
            </div>
            
            <div class="catalog-sidebar">
                <h2 class="section-title"><i class="fas fa-clipboard-check"></i> Current Status</h2>
                
                <div class="current-status">
                    <div class="status-item">
                        <span class="status-label">Repository</span>
                        <span class="status-value">cloudera-ai-demo-hub</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Catalog System</span>
                        <span class="status-value status-complete">✓ Implemented</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Folder Structure</span>
                        <span class="status-value status-complete">✓ Business Scenarios</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Current Phase</span>
                        <span class="status-value status-complete">Phase 1 Complete</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Multi-tagging</span>
                        <span class="status-value status-current">Active</span>
                    </div>
                </div>
                
                <h2 class="section-title"><i class="fas fa-sitemap"></i> Folder Structure</h2>
                <div class="folder-structure">
                    <div class="folder">cloudera-ai-demo-hub/</div>
                    <div class="folder indent-1">demos/</div>
                    <div class="folder indent-2">01_AI_Agents_Copilots/</div>
                    <div class="folder indent-3">customer-service-chatbot/</div>
                    <div class="folder indent-3">internal-knowledge-copilot/</div>
                    <div class="folder indent-2">02_AI_Governance_Guardrails/</div>
                    <div class="folder indent-3">llm-content-safety-filter/</div>
                    <div class="folder indent-3">model-bias-fairness-check/</div>
                    <div class="folder indent-2">03_Document_AI_Automation/</div>
                    <div class="folder indent-3">invoice-processing-pipeline/</div>
                    <div class="folder indent-3">contract-review-analyzer/</div>
                    <div class="folder indent-2">04_Predictive_ML_Operations/</div>
                    <div class="folder indent-3">predictive-maintenance/</div>
                    <div class="folder indent-3">real-time-fraud-scoring/</div>
                    <div class="file indent-1">catalog.yml</div>
                    <div class="file indent-1">README.md</div>
                </div>
                
                <h2 class="section-title"><i class="fas fa-tags"></i> Multi-Tagging System</h2>
                <div class="tag-system">
                    <p>Each demo can have <span class="highlight">multiple tags</span> across dimensions:</p>
                    <div style="margin: 15px 0;">
                        <span class="tag-category">scenario</span>
                        <span class="tag-category">capabilities</span>
                        <span class="tag-category">industry</span>
                        <span class="tag-category">use_case</span>
                    </div>
                    <p>Example: An LLM Governance demo for banking could be tagged with:</p>
                    <p><span class="highlight">AI_Governance_Guardrails</span> + <span class="highlight">LLM Applications</span> + <span class="highlight">financial-services</span> + <span class="highlight">compliance</span></p>
                </div>
                
                <h2 class="section-title"><i class="fas fa-code"></i> catalog.yml Preview</h2>
                <div class="catalog-preview">
                    <span class="yaml-key">demos</span>:<br>
                    <span class="yaml-comment">  # Sample demo entry with multi-tagging</span><br>
      - <span class="yaml-key">id</span>: <span class="yaml-string">guardrails_finance_01</span><br>
        <span class="yaml-key">name</span>: <span class="yaml-string">"LLM Content Safety for Banking"</span><br>
        <span class="yaml-key">path</span>: <span class="yaml-string">"demos/02_AI_Governance_Guardrails/llm-content-safety-filter"</span><br>
        <span class="yaml-key">scenario</span>: <span class="yaml-string">"AI_Governance_Guardrails"</span><br>
        <span class="yaml-key">capabilities</span>:<br>
          - <span class="yaml-string">"Large Language Model (LLM) Applications"</span><br>
          - <span class="yaml-string">"AI Guardrails & Governance"</span><br>
        <span class="yaml-key">industry</span>: <span class="yaml-string">"financial-services"</span><br>
        <span class="yaml-key">use_case</span>: <span class="yaml-string">"compliance-automation"</span><br>
        <span class="yaml-key">owner</span>: <span class="yaml-string">"@demo_owner"</span>
                </div>
            </div>
        </div>
        
        <footer>
            <p><strong>Cloudera AI Demo Hub Cataloging System</strong> | Business scenario organization with multi-dimensional tagging</p>
            <p>Phase 1 complete: Repository structured, catalog.yml implemented, ready for demo submissions</p>
            <p style="margin-top: 10px; color: #ed8936; font-weight: 600;">
                <i class="fas fa-lightbulb"></i> Key Insight: Organize by business problems, tag for multiple discoverability paths
            </p>
        </footer>
    </div>

    <script>
        // Initialize Mermaid
        mermaid.initialize({
            startOnLoad: true,
            theme: 'neutral',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            },
            themeCSS: `
                .node rect {
                    fill: #ffffff;
                    stroke: #ed8936;
                    stroke-width: 2px;
                }
                .node.clickable rect {
                    fill: #fffaf0;
                }
                .cluster rect {
                    fill: #f8fafc;
                    stroke: #cbd5e0;
                    stroke-width: 2px;
                    rx: 8;
                    ry: 8;
                }
                .cluster-label {
                    fill: #2d3748;
                    font-weight: bold;
                }
                .edgeLabel {
                    background-color: #f8fafc;
                    fill: #f8fafc;
                }
                .edgeLabel rect {
                    fill: #f8fafc;
                }
                .label {
                    color: #2d3748;
                }
                .node.completed rect {
                    fill: #c6f6d5;
                    stroke: #38a169;
                }
                .node.current rect {
                    fill: #fed7d7;
                    stroke: #ed8936;
                }
                .node.pending rect {
                    fill: #bee3f8;
                    stroke: #4299e1;
                }
                .edgePath .path {
                    stroke: #a0aec0;
                }
                .arrowheadPath {
                    fill: #a0aec0;
                }
            `
        });
        
        // Catalog-focused flowchart definition
        const flowchartDefinition = `
        graph TD
            A["Start: GitHub Account"] --> B
            
            subgraph B [Phase 1: Foundation & Structure - COMPLETE]
                B1["Create Repository<br>cloudera-ai-demo-hub"] --> B2["Create Business Scenario Folders"]
                B2 --> B3["Implement catalog.yml<br>with Multi-tagging"]
                B3 --> B4["Add README Templates<br>Standardize Documentation"]
            end
            
            B --> C{"Catalog Decision:<br>How to organize demos?"}
            C -- "Business Scenarios" --> D

            subgraph D [Selected Path: Business Organization]
                D1["01_AI_Agents_Copilots"] --> D2["02_AI_Governance_Guardrails"]
                D2 --> D3["03_Document_AI_Automation"]
                D3 --> D4["04_Predictive_ML_Operations"]
            end
            
            D --> F
            
            subgraph F [Phase 2: Multi-Dimensional Tagging]
                F1["Tag by Scenario<br>(matches folder)"] --> F2["Tag by Capabilities<br>(multi-select)"]
                F2 --> F3["Tag by Industry"] 
                F3 --> F4["Tag by Use Case"]
            end
            
            F --> G
            
            subgraph G [Phase 3: Launch & Discovery]
                G1["Teams Submit Demos<br>with catalog.yml entries"] --> G2["Catalog becomes<br>searchable index"]
                G2 --> G3["Sales: Filter by Industry<br>Engineers: Filter by Capabilities"]
            end
            
            G --> H["Operational Demo Hub<br>Smart Catalog System Active"]
            
            %% Styling
            class A,B1,B2,B3,B4 completed
            class C current
            class D1,D2,D3,D4,F1,F2,F3,F4,G1,G2,G3,H pending
        `;
        
        // Render initial flowchart
        document.getElementById('flowchart').innerHTML = `<div class="mermaid">${flowchartDefinition}</div>`;
        
        // Function to show phase details
        function showPhase(phase) {
            let message = "";
            
            switch(phase) {
                case 'phase1':
                    message = "Phase 1 Complete: Repository created with business scenario folders (01_AI_Agents_Copilots, 02_AI_Governance_Guardrails, etc.) and catalog.yml implemented.";
                    break;
                case 'phase2':
                    message = "Cataloging System: Each demo can have multiple tags - scenario (folder), capabilities (multi-select), industry, use_case. This enables filtering from different perspectives.";
                    break;
                case 'phase3':
                    message = "Launch Process: Teams add demos with catalog.yml entries. The catalog becomes a searchable index for Sales (by industry) and Engineers (by capabilities).";
                    break;
            }
            
            if (message) {
                alert(message);
            }
        }
        
        // Initialize
        mermaid.init();
    </script>
</body>
</html>