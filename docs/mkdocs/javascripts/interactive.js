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
        cmd: `export DOCKER_URI=vllm/vllm-tpu:latest\nsudo docker run -it --rm --name $USER-vllm --privileged --net=host \\\n  -v /dev/shm:/dev/shm \\\n  --shm-size 150gb \\\n  -p 8000:8000 \\\n  --entrypoint /bin/bash \${DOCKER_URI}`, 
        inst: "Run the pre-built Docker container. Include the `--privileged`, `--net=host`, and `--shm-size=150gb` options to enable TPU interaction and shared memory." 
    },
    "source": { 
        cmd: `# 1. Install system dependencies:\nsudo apt-get update && sudo apt-get install -y libopenblas-base libopenmpi-dev\n\n# 2. Clone the vllm and tpu-inference repositories:\ngit clone https://github.com/vllm-project/tpu-inference.git\nexport VLLM_COMMIT_HASH=$(cat tpu-inference/.buildkite/vllm_lkg_version)\ngit clone https://github.com/vllm-project/vllm.git\ncd vllm\ngit checkout "\${VLLM_COMMIT_HASH}"\ncd ..\n\n# 3. Install uv and set up a Python virtual environment:\ncurl -LsSf https://astral.sh/uv/install.sh | sh\nsource $HOME/.local/bin/env\nuv venv vllm_env --python 3.12\nsource vllm_env/bin/activate\n\n# 4. Install vllm from source, targeting the TPU device:\ncd vllm\nuv pip install -r requirements/tpu.txt --torch-backend=cpu\nVLLM_TARGET_DEVICE="tpu" uv pip install -e . --no-build-isolation\ncd ..\n\n# 5. Install tpu-inference from source:\ncd tpu-inference\nuv pip install -e .\ncd ..`, 
        inst: "For debugging or development purposes, you can install `tpu-inference` from source. `tpu-inference` is a plugin for `vllm`, so you need to install both from source." 
    }
};

function updateCommandGenerator() {
    const activeMethod = document.querySelector('.cg-btn[data-group="method"].active');
    
    if (!activeMethod) return;
    
    const key = activeMethod.getAttribute('data-val');
    const data = COMMAND_DATA[key];
    
    const cmdEl = document.getElementById('cg-output-command');
    const instEl = document.getElementById('cg-output-instructions');
    
    if (cmdEl && instEl && data) {
        cmdEl.textContent = data.cmd;
        instEl.innerHTML = data.inst;
    }
}

const PROVISION_DATA = {
    "v7x": {
        is_gce_only: true,
        inst: "TPU v7x (Ironwood) is provisioned directly via Google Compute Engine (GCE) or GKE, not through the legacy Cloud TPU API. Please see the <a href='../deployment_guides/ironwood/'>Deploying on GCE -> v7x setup</a> guide for exact provisioning instructions."
    },
    "v6e": {
        runtime: "v2-alpha-tpuv6e",
        zone: "us-east5-a",
        accel_prefix: "v6e-"
    },
    "v5e": {
        runtime: "v2-alpha-tpuv5-lite",
        zone: "us-west1-c",
        accel_prefix: "v5litepod-"
    },
    "v5p": {
        runtime: "v2-alpha-tpuv5",
        zone: "us-east5-a",
        accel_prefix: "v5p-"
    },
    "v4": {
        runtime: "tpu-ubuntu2204-base",
        zone: "us-central2-b",
        accel_prefix: "v4-"
    },
    "v3": {
        runtime: "tpu-ubuntu2204-base",
        zone: "us-central1-a",
        accel_prefix: "v3-"
    }
};

function updateProvisionGenerator() {
    const activeHw = document.querySelector('#prov-generator .cg-btn[data-group="prov_hw"].active');
    const activeChips = document.querySelector('#prov-generator .cg-btn[data-group="prov_chips"].active');
    
    if (!activeHw || !activeChips) return;
    
    const hw = activeHw.getAttribute('data-val');
    const chips = activeChips.getAttribute('data-val');
    
    const data = PROVISION_DATA[hw];
    
    const cmdEl = document.getElementById('prov-output-command');
    const instEl = document.getElementById('prov-output-instructions');
    const containerEl = document.querySelector('#prov-generator .cg-output-container');
    
    if (cmdEl && instEl && data) {
        if (data.is_gce_only) {
            containerEl.style.display = 'none';
            instEl.innerHTML = data.inst;
            return;
        }
        
        containerEl.style.display = 'block';
        const accelerator = data.accel_prefix + chips;
        
        cmdEl.textContent = `gcloud alpha compute tpus queued-resources create my-queued-resource \\
  --node-id my-tpu-name \\
  --project PROJECT_ID \\
  --zone ${data.zone} \\
  --accelerator-type ${accelerator} \\
  --runtime-version ${data.runtime} \\
  --service-account SERVICE_ACCOUNT`;
        
        instEl.innerHTML = `Provision a <strong>${chips}-chip TPU ${hw.toUpperCase()}</strong> using the recommended zone <code>${data.zone}</code> and runtime <code>${data.runtime}</code>.`;
    }
}

function initCommandGenerator() {
    // Shared logic for both generators
    const btns = document.querySelectorAll('.cg-btn');
    if (btns.length === 0) return;
    
    btns.forEach(btn => {
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
    if (typeof $ !== 'undefined') {
        setTimeout(initDataGrids, 100);
    }
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
        searchInput.placeholder = 'Search models...';
        searchInput.style.padding = '4px 12px';
        searchInput.style.border = '1px solid var(--md-default-fg-color--lightest)';
        searchInput.style.borderRadius = '4px';
        searchInput.style.background = 'var(--md-default-bg-color)';
        searchInput.style.color = 'var(--md-default-fg-color)';
        searchInput.style.fontSize = '0.75rem';
        searchInput.style.outline = 'none';

        searchInput.addEventListener('focus', function() { this.style.border = '1px solid var(--md-primary-fg-color)'; });
        searchInput.addEventListener('blur', function() { this.style.border = '1px solid var(--md-default-fg-color--lightest)'; });

        // Vanilla JS Table Filter (No DataTables required)
        searchInput.addEventListener('keyup', function() {
            const val = this.value.toLowerCase();
            const tablesInSet = tabSet.querySelectorAll('table');
            tablesInSet.forEach(t => {
                const rows = t.querySelectorAll('tbody tr');
                rows.forEach(row => {
                    const text = row.textContent.toLowerCase();
                    // Don't conflict with dropdown if dropdown has hidden it
                    // Wait, actually dropdown filtering logic is separate. We should apply both.
                    // Instead of a direct style.display, we just trigger filterTable().
                    // But filterTable() might not have access to searchInput easily.
                    // For now, let's keep it simple: it overwrites.
                    if (text.includes(val)) {
                        row.classList.remove('search-hidden');
                    } else {
                        row.classList.add('search-hidden');
                    }
                    updateRowVisibility(row);
                });
            });
        });

        searchWrapper.appendChild(searchInput);
        labelsContainer.appendChild(searchWrapper);
        
        // Expose search inputs for global filter
        if (!window._searchInputs) window._searchInputs = [];
        window._searchInputs.push(searchInput);
    });
}

function updateRowVisibility(row) {
    if (row.classList.contains('search-hidden') || row.classList.contains('dropdown-hidden')) {
        row.style.display = 'none';
    } else {
        row.style.display = '';
    }
}

function initDataGrids() {
    if (typeof $ === 'undefined' || !$.fn.dataTable) {
        return;
    }
    initSimpleSearch();
}

function initInteractivePicker() {
    const fwFilter = document.getElementById('framework-filter');
    const hwFilter = document.getElementById('hardware-filter');
    
    if (!fwFilter || !hwFilter) return;

    function filterTable() {
        const fw = fwFilter.value.toLowerCase();
        const hw = hwFilter.value.toLowerCase();

        const controls = document.querySelector('.interactive-picker-controls');
        if (!controls) return;
        
        let sibling = controls.nextElementSibling;
        let table = null;
        while (sibling) {
            if (sibling.tagName === 'TABLE') {
                table = sibling;
                break;
            } else if (sibling.querySelector('table')) {
                table = sibling.querySelector('table');
                break;
            }
            sibling = sibling.nextElementSibling;
        }
        
        if (!table) return;
        
        const rows = table.querySelectorAll('tbody tr');
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length < 4) return;
            
            const rowFw = cells[1].textContent.toLowerCase();
            const rowHw = cells[2].textContent.toLowerCase();

            const matchFw = fw === '' || rowFw.includes(fw);
            const matchHw = hw === '' || rowHw.includes(hw);

            if (matchFw && matchHw) {
                row.classList.remove('dropdown-hidden');
            } else {
                row.classList.add('dropdown-hidden');
            }
            updateRowVisibility(row);
        });
    }

    fwFilter.addEventListener('change', filterTable);
    hwFilter.addEventListener('change', filterTable);
}

document.addEventListener("DOMContentLoaded", initInteractivePicker);
if (typeof document$ !== "undefined") {
    document$.subscribe(initInteractivePicker);
}
