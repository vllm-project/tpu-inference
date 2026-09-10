/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Interactive UI Components Logic */

const COMMAND_DATA = {
    "uv_pip": { 
        cmd: `# 1. Create a working directory:\nmkdir ~/work-dir\ncd ~/work-dir\n\n# 2. Install uv and set up a Python virtual environment:\n# If you prefer standard pip, simply use \`python3.12 -m venv vllm_env\`\ncurl -LsSf https://astral.sh/uv/install.sh | sh\nsource $HOME/.local/bin/env\nuv venv vllm_env --python 3.12\nsource vllm_env/bin/activate\n\n# 3. Use the following command to install vllm-tpu using uv or pip:\nuv pip install vllm-tpu\n# Or instead: pip install vllm-tpu`, 
        inst: "Install the latest official release using uv pip for any supported TPU." 
    },
    "docker": { 
        "latest": {
            cmd: `export DOCKER_URI=vllm/vllm-tpu:latest\nsudo docker run -it --rm --name $USER-vllm --privileged --net=host \\\n  -v /dev/shm:/dev/shm \\\n  --shm-size 150gb \\\n  -p 8000:8000 \\\n  --entrypoint /bin/bash \${DOCKER_URI}`, 
            inst: "Run the official <strong>release</strong> Docker container (<code>vllm/vllm-tpu:latest</code>). Include the <code>--privileged</code>, <code>--net=host</code>, and <code>--shm-size=150gb</code> options to enable TPU interaction and shared memory." 
        },
        "nightly": {
            cmd: `export DOCKER_URI=vllm/vllm-tpu:nightly\nsudo docker run -it --rm --name $USER-vllm --privileged --net=host \\\n  -v /dev/shm:/dev/shm \\\n  --shm-size 150gb \\\n  -p 8000:8000 \\\n  --entrypoint /bin/bash \${DOCKER_URI}`, 
            inst: "Run the pre-built <strong>nightly</strong> Docker container (<code>vllm/vllm-tpu:nightly</code>) containing the latest development changes. Include the <code>--privileged</code>, <code>--net=host</code>, and <code>--shm-size=150gb</code> options to enable TPU interaction and shared memory." 
        }
    },
    "source": { 
        cmd: `# 1. Install system dependencies:\nsudo apt-get update && sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev\n\n# 2. Clone the vllm and tpu-inference repositories:\ngit clone https://github.com/vllm-project/tpu-inference.git\nexport VLLM_COMMIT_HASH=$(cat tpu-inference/.buildkite/vllm_lkg.version)\ngit clone https://github.com/vllm-project/vllm.git\ncd vllm\ngit checkout "\${VLLM_COMMIT_HASH}"\ncd ..\n\n# 3. Install uv and set up a Python virtual environment:\ncurl -LsSf https://astral.sh/uv/install.sh | sh\nsource $HOME/.local/bin/env\nuv venv vllm_env --python 3.12\nsource vllm_env/bin/activate\n\n# 4. Install vllm from source, targeting the TPU device:\ncd vllm\nuv pip install -r requirements/tpu.txt --torch-backend=cpu\nVLLM_TARGET_DEVICE="tpu" uv pip install -e . --no-build-isolation\ncd ..\n\n# 5. Install tpu-inference from source:\ncd tpu-inference\nuv pip install -e .\ncd ..`, 
        inst: "For debugging or development purposes, you can install <code>tpu-inference</code> from source. <code>tpu-inference</code> is a plugin for <code>vllm</code>, so you need to install both from source." 
    }
};

/* Minimal shell highlighter for dynamically injected snippets. Build-time
   Pygments cannot color runtime content, so wrap comments, strings, and
   variables in Pygments token classes styled in interactive.css. */
function highlightShell(code) {
    const escaped = code
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    const tokenRegex = /("[^"]*"|'[^']*'|\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|--[a-zA-Z0-9_-]+|-[a-zA-Z0-9]+|&&|\|\||\||\\|\b(?:export|source|sudo|cd|mkdir|git|uv|pip|curl|gcloud|docker|apt-get|python|python3\.12)\b|\b(?:install|run|clone|checkout|update|venv|create|compute|tpus|queued-resources|tpu-vm|ssh|alpha)\b)/g;
    return escaped.split('\n').map(line => {
        if (/^\s*#/.test(line)) {
            return '<span class="c1">' + line + '</span>';
        }
        return line.replace(tokenRegex, match => {
            if (match.startsWith('"') || match.startsWith("'")) {
                return '<span class="s2">' + match + '</span>';
            }
            if (match.startsWith('$')) {
                return '<span class="nv">' + match + '</span>';
            }
            if (match.startsWith('-')) {
                return '<span class="na">' + match + '</span>';
            }
            if (match === '&&' || match === '||' || match === '|' || match === '\\') {
                return '<span class="o">' + match + '</span>';
            }
            if (/^(?:export|source|sudo|cd|mkdir|git|uv|pip|curl|gcloud|docker|apt-get|python|python3\.12)$/.test(match)) {
                return '<span class="k">' + match + '</span>';
            }
            return '<span class="nb">' + match + '</span>';
        });
    }).join('\n');
}

function updateCommandGenerator() {
    const activeMethod = document.querySelector('.cg-btn[data-group="method"].active');
    if (!activeMethod) return;

    const method = activeMethod.getAttribute('data-val');
    const dockerImgGroup = document.getElementById('docker-image-group');

    let data = COMMAND_DATA[method];

    if (method === 'docker') {
        if (dockerImgGroup) dockerImgGroup.style.display = 'flex';
        const activeImgBtn = document.querySelector('.cg-btn[data-group="docker_img"].active');
        const tag = activeImgBtn ? activeImgBtn.getAttribute('data-val') : 'latest';
        data = COMMAND_DATA.docker[tag] || COMMAND_DATA.docker.latest;
    } else {
        if (dockerImgGroup) dockerImgGroup.style.display = 'none';
    }

    const cmdEl = document.getElementById('cg-output-command');
    const instEl = document.getElementById('cg-output-instructions');

    if (cmdEl && instEl && data) {
        cmdEl.innerHTML = highlightShell(data.cmd);
        instEl.innerHTML = data.inst;
    }
}

const PROVISION_DATA = {
    "v6e": {
        runtime: "v2-alpha-tpuv6e",
        zone: "us-east5-a",
        accel_prefix: "v6e-",
        supports_flex: true,
        valid_chips: [1, 4, 8]
    },
    "v5e": {
        runtime: "v2-alpha-tpuv5-lite",
        zone: "us-west1-c",
        accel_prefix: "v5litepod-",
        supports_flex: true,
        valid_chips: [1, 4, 8, 16, 32, 64]
    },
    "v5p": {
        runtime: "v2-alpha-tpuv5",
        zone: "us-east5-a",
        accel_prefix: "v5p-",
        uses_core_count: true,
        supports_flex: true,
        valid_chips: [4]
    },
    "v4": {
        runtime: "tpu-ubuntu2204-base",
        zone: "us-central2-b",
        accel_prefix: "v4-",
        uses_core_count: true,
        supports_flex: false,
        valid_chips: [4, 8, 16, 32, 64]
    },
    "v3": {
        runtime: "tpu-ubuntu2204-base",
        zone: "us-central1-a",
        accel_prefix: "v3-",
        uses_core_count: true,
        supports_flex: false,
        valid_chips: [4, 16, 32, 64]
    }
};

function updateProvisionGenerator() {
    const activeHw = document.querySelector('#prov-generator .cg-btn[data-group="prov_hw"].active');
    const activeChips = document.querySelector('#prov-generator .cg-btn[data-group="prov_chips"].active');
    const activeModel = document.querySelector('#prov-generator .cg-btn[data-group="prov_model"].active');
    
    if (!activeHw || !activeChips) return;
    
    const hw = activeHw.getAttribute('data-val');
    let chips = activeChips.getAttribute('data-val');
    let model = activeModel ? activeModel.getAttribute('data-val') : 'standard';
    
    const data = PROVISION_DATA[hw];
    
    const cmdEl = document.getElementById('prov-output-command');
    const instEl = document.getElementById('prov-output-instructions');
    const containerEl = document.querySelector('#prov-generator .cg-output-container');
    
    if (cmdEl && instEl && data) {
        // Grey out invalid chip counts for this hardware
        if (data.valid_chips) {
            document.querySelectorAll('#prov-generator .cg-btn[data-group="prov_chips"]').forEach(btn => {
                const val = parseInt(btn.getAttribute('data-val'), 10);
                if (data.valid_chips.includes(val)) {
                    btn.classList.remove('disabled');
                    btn.removeAttribute('disabled');
                } else {
                    btn.classList.add('disabled');
                    btn.setAttribute('disabled', 'true');
                    if (btn.classList.contains('active')) {
                        btn.classList.remove('active');
                        btn.setAttribute('aria-pressed', 'false');
                        const firstValidBtn = document.querySelector(`#prov-generator .cg-btn[data-group="prov_chips"][data-val="${data.valid_chips[0]}"]`);
                        if (firstValidBtn) {
                            firstValidBtn.classList.add('active');
                            firstValidBtn.setAttribute('aria-pressed', 'true');
                            chips = String(data.valid_chips[0]);
                        }
                    }
                }
            });
        }

        // Grey out Flex-start if unsupported on this hardware
        const flexBtn = document.querySelector('#prov-generator .cg-btn[data-group="prov_model"][data-val="flex_start"]');
        if (flexBtn) {
            if (data.supports_flex) {
                flexBtn.classList.remove('disabled');
                flexBtn.removeAttribute('disabled');
            } else {
                flexBtn.classList.add('disabled');
                flexBtn.setAttribute('disabled', 'true');
                if (model === 'flex_start') {
                    const stdBtn = document.querySelector('#prov-generator .cg-btn[data-group="prov_model"][data-val="standard"]');
                    if (stdBtn) {
                        flexBtn.classList.remove('active');
                        flexBtn.setAttribute('aria-pressed', 'false');
                        stdBtn.classList.add('active');
                        stdBtn.setAttribute('aria-pressed', 'true');
                        model = 'standard';
                    }
                }
            }
        }

        containerEl.style.display = 'block';

        if (hw === 'v6e' || hw === 'v5p') {
            const machineType = hw === 'v6e' ? `ct6e-standard-${chips}t` : `ct5p-hightpu-${chips}t`;
            const gceNote = hw === 'v6e' ? ' <em>(Note: 16+ chip topologies require multi-host MIGs or GKE)</em>' : ' <em>(Note: 8+ chip topologies require multi-host MIGs or GKE)</em>';
            if (model === 'flex_start') {
                cmdEl.innerHTML = highlightShell(`gcloud compute instances create my-tpu-vm \\
  --zone=ZONE \\
  --machine-type=${machineType} \\
  --provisioning-model=FLEX_START \\
  --request-valid-for-duration=2h \\
  --max-run-duration=4h \\
  --instance-termination-action=DELETE \\
  --image-project=ubuntu-os-accelerator-images \\
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \\
  --maintenance-policy=TERMINATE`);
                instEl.innerHTML = `Provision a discounted <strong>${chips}-chip TPU ${hw.toUpperCase()}</strong> VM instance using Compute Engine (GCE) with Flex-start (DWS). Replace <code>ZONE</code> with your target zone.${gceNote}`;
            } else {
                cmdEl.innerHTML = highlightShell(`gcloud compute instances create my-tpu-vm \\
  --zone=ZONE \\
  --machine-type=${machineType} \\
  --image-project=ubuntu-os-accelerator-images \\
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \\
  --maintenance-policy=TERMINATE`);
                instEl.innerHTML = `Provision a standard <strong>${chips}-chip TPU ${hw.toUpperCase()}</strong> VM instance using Compute Engine (GCE). Replace <code>ZONE</code> with your target zone.${gceNote}`;
            }
        } else {
            let accelerator;
            let coreNote = "";
            if (data.uses_core_count) {
                const cores = parseInt(chips, 10) * 2;
                accelerator = data.accel_prefix + cores;
                coreNote = ` (${cores} TensorCores)`;
                if (parseInt(chips, 10) === 1) {
                    coreNote += ` — <em>Note: minimum standard slice for ${hw.toUpperCase()} is 4 chips (${data.accel_prefix}8)</em>`;
                }
            } else {
                accelerator = data.accel_prefix + chips;
            }

            if (model === 'flex_start') {
                cmdEl.innerHTML = highlightShell(`gcloud alpha compute tpus queued-resources create my-queued-resource \\
  --node-id my-tpu-name \\
  --project PROJECT_ID \\
  --zone ZONE \\
  --accelerator-type ${accelerator} \\
  --runtime-version ${data.runtime} \\
  --provisioning-model flex-start \\
  --max-run-duration 4h \\
  --valid-until-duration 4h \\
  --service-account SERVICE_ACCOUNT`);
                instEl.innerHTML = `Provision a discounted <strong>${chips}-chip TPU ${hw.toUpperCase()}${coreNote}</strong> using Flex-start with the legacy Queued Resources API. Replace <code>PROJECT_ID</code>, <code>ZONE</code>, and <code>SERVICE_ACCOUNT</code> with your values.`;
            } else {
                cmdEl.innerHTML = highlightShell(`gcloud alpha compute tpus queued-resources create my-queued-resource \\
  --node-id my-tpu-name \\
  --project PROJECT_ID \\
  --zone ZONE \\
  --accelerator-type ${accelerator} \\
  --runtime-version ${data.runtime} \\
  --service-account SERVICE_ACCOUNT`);
                instEl.innerHTML = `Provision a standard <strong>${chips}-chip TPU ${hw.toUpperCase()}${coreNote}</strong> using the legacy Queued Resources API. Replace <code>PROJECT_ID</code>, <code>ZONE</code>, and <code>SERVICE_ACCOUNT</code> with your values.`;
            }
        }
    }
}

function initCommandGenerator() {
    // Shared logic for both generators
    const btns = document.querySelectorAll('.cg-btn');
    if (btns.length === 0) return;
    
    btns.forEach(btn => {
        if (btn.dataset.cgInitialized === 'true') return;
        btn.dataset.cgInitialized = 'true';

        btn.addEventListener('click', function() {
            const group = this.getAttribute('data-group');
            // Remove active from others in group
            const container = this.closest('.command-generator-container');
            container.querySelectorAll(`.cg-btn[data-group="${group}"]`).forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-pressed', 'false');
            });
            // Add to clicked
            this.classList.add('active');
            this.setAttribute('aria-pressed', 'true');
            
            if (group.startsWith('prov_')) {
                updateProvisionGenerator();
            } else {
                updateCommandGenerator();
            }
        });
    });
    
    updateCommandGenerator();
    updateProvisionGenerator();
}

function initInteractiveComponents() {
    initCommandGenerator();
    initSimpleSearch();
}

document.addEventListener("DOMContentLoaded", function() {
    initInteractiveComponents();
});

if (typeof document$ !== "undefined") {
    document$.subscribe(function() {
        initInteractiveComponents();
    });
}

function initSimpleSearch() {
    const tabSets = document.querySelectorAll('.tabbed-set');
    if (tabSets.length === 0) return;

    tabSets.forEach(tabSet => {
        if (!tabSet.querySelector('table')) return;
        const labelsContainer = tabSet.querySelector('.tabbed-labels');
        if (!labelsContainer || labelsContainer.querySelector('.custom-tab-search')) return;

        const searchWrapper = document.createElement('div');
        searchWrapper.className = 'custom-tab-search';
        searchWrapper.style.marginLeft = 'auto'; 
        searchWrapper.style.display = 'flex';
        searchWrapper.style.alignItems = 'center';
        searchWrapper.style.paddingRight = '8px';

        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = 'Search...';
        searchInput.style.padding = '4px 12px';
        searchInput.style.border = '1px solid var(--md-default-fg-color--lightest)';
        searchInput.style.borderRadius = '4px';
        searchInput.style.background = 'var(--md-default-bg-color)';
        searchInput.style.color = 'var(--md-default-fg-color)';
        searchInput.style.fontSize = '0.75rem';
        searchInput.style.outline = 'none';

        searchInput.addEventListener('focus', function() { this.style.border = '1px solid var(--md-primary-fg-color)'; });
        searchInput.addEventListener('blur', function() { this.style.border = '1px solid var(--md-default-fg-color--lightest)'; });

        searchInput.addEventListener('input', function() {
            const val = this.value.toLowerCase();
            const tablesInSet = tabSet.querySelectorAll('table');
            tablesInSet.forEach(t => {
                const rows = t.querySelectorAll('tbody tr');
                rows.forEach(row => {
                    const text = row.textContent.toLowerCase();
                    row.style.display = text.includes(val) ? '' : 'none';
                });
            });
        });

        searchWrapper.appendChild(searchInput);
        labelsContainer.appendChild(searchWrapper);
    });
}
