(function () {
  var math = window.LaunchMath;
  var THREE = window.THREE;
  if (!math || !THREE) {
    return;
  }

  var statsRoot = document.getElementById("stats");
  var controlsRoot = document.getElementById("control-groups");
  var combinedInfoRoot = document.getElementById("combined-info");
  var controlsHintRoot = document.getElementById("controls-hint");
  var combinedViewRoot = document.getElementById("combined-view");
  var combinedViewportShellRoot = document.getElementById("combined-viewport-shell");
  var hierarchyRoot = document.getElementById("hierarchy-view");
  var gpuUsagePanelRoot = document.getElementById("gpu-usage-panel");
  var threadModeButton = document.getElementById("mode-thread");
  var warpModeButton = document.getElementById("mode-warp");
  var organizeBySmToggle = document.getElementById("organize-by-sm");
  var showSmMemoryToggle = document.getElementById("show-sm-memory");
  var resetViewButton = document.getElementById("reset-view");
  var lastValidConfig = null;

  var state = {
    config: {
      grid: { x: 3, y: 2, z: 2 },
      block: { x: 4, y: 4, z: 2 },
      kernelUsage: {
        gpuKey: "H100",
        registersPerThread: 37,
        sharedMemoryPerBlockKB: 8
      }
    },
    hovered: null,
    selectedBlock: null,
    colorMode: "thread",
    visualBudget: {
      tier: "high",
      maxThreads: math.MAX_TOTAL_THREADS,
      baselineMaxThreads: math.MAX_TOTAL_THREADS,
      label: "Global cap"
    },
    performance: {
      recentRenderMs: [],
      slowRenderCount: 0,
      reducingBudget: false
    },
    viewOptions: {
      organizeBySm: true,
      showSmMemory: true
    }
  };

  var controlSchema = [
    {
      title: "Grid Size",
      prefix: "grid",
      inputType: "range",
      help: "How many blocks to dispatch on X, Y, Z. Grid sliders go to " + math.MAX_GRID_DIMENSION + ", but total launched threads cannot exceed " + math.MAX_TOTAL_THREADS + ".",
      fields: [
        ["x", "Grid X", 1, math.MAX_GRID_DIMENSION],
        ["y", "Grid Y", 1, math.MAX_GRID_DIMENSION],
        ["z", "Grid Z", 1, math.MAX_GRID_DIMENSION]
      ]
    },
    {
      title: "Block Size",
      prefix: "block",
      inputType: "range-pow2",
      help: "How many threads each block contains on X, Y, Z. Block sliders move in powers of 2 up to the CUDA-style dimension caps, and total launched threads cannot exceed " + math.MAX_TOTAL_THREADS + ".",
      fields: [
        ["x", "Block X", 1, math.MAX_BLOCK_DIMENSION.x],
        ["y", "Block Y", 1, math.MAX_BLOCK_DIMENSION.y],
        ["z", "Block Z", 1, math.MAX_BLOCK_DIMENSION.z]
      ]
    }
  ];

  function getCurrentGpuProfile() {
    return math.GPU_PROFILES[state.config.kernelUsage.gpuKey];
  }

  function normalizeBlockControlState() {
    state.config.block.x = math.snapToPowerOfTwo(state.config.block.x, math.MAX_BLOCK_DIMENSION.x);
    state.config.block.y = math.snapToPowerOfTwo(state.config.block.y, math.MAX_BLOCK_DIMENSION.y);
    state.config.block.z = math.snapToPowerOfTwo(state.config.block.z, math.MAX_BLOCK_DIMENSION.z);
  }

  function getVisualizationThreadBudget() {
    var multiplier = state.viewOptions.organizeBySm ? 1 : 3;
    return Math.min(math.MAX_TOTAL_THREADS, state.visualBudget.maxThreads * multiplier);
  }

  function detectLocalVisualizationBudget() {
    var canvas = document.createElement("canvas");
    var gl = canvas.getContext("webgl2") || canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    var deviceInfo = {
      webglAvailable: Boolean(gl),
      hardwareConcurrency: navigator.hardwareConcurrency || 1,
      vendor: "",
      renderer: ""
    };

    if (gl) {
      var debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
      if (debugInfo) {
        deviceInfo.vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || "";
        deviceInfo.renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || "";
      } else {
        deviceInfo.vendor = gl.getParameter(gl.VENDOR) || "";
        deviceInfo.renderer = gl.getParameter(gl.RENDERER) || "";
      }
    }

    state.visualBudget = math.recommendVisualizationBudget(deviceInfo);
    state.visualBudget.baselineMaxThreads = state.visualBudget.maxThreads;
  }

  function getKernelUsageForLaunch(launch) {
    return {
      block: launch.block,
      registersPerThread: state.config.kernelUsage.registersPerThread,
      sharedMemoryPerBlockKB: state.config.kernelUsage.sharedMemoryPerBlockKB
    };
  }

  function getCurrentPacking(launch) {
    return math.computeSmPacking(launch, getKernelUsageForLaunch(launch), getCurrentGpuProfile());
  }

  function getCurrentResidency(launch) {
    return math.computeGpuResidency(launch, getKernelUsageForLaunch(launch), getCurrentGpuProfile());
  }

  function getCurrentValidation(launch) {
    return math.validateLaunchConfig(launch, getCurrentGpuProfile());
  }

  function getDerivedState(launch) {
    return {
      gpu: getCurrentGpuProfile(),
      residency: getCurrentResidency(launch),
      validation: getCurrentValidation(launch)
    };
  }

  function getCurrentKernelUsageLimits() {
    return math.computeKernelUsageLimits(state.config.block, getCurrentGpuProfile());
  }

  function clampKernelUsageToGpuLimits() {
    var limits = getCurrentKernelUsageLimits();
    state.config.kernelUsage.registersPerThread = Math.min(
      limits.maxRegistersPerThread,
      Math.max(0, Number(state.config.kernelUsage.registersPerThread) || 0)
    );
    state.config.kernelUsage.sharedMemoryPerBlockKB = Math.min(
      limits.maxSharedMemoryPerBlockKB,
      Math.max(0, Number(state.config.kernelUsage.sharedMemoryPerBlockKB) || 0)
    );
    return limits;
  }

  function fmt3(v) {
    return vectorHtml(v);
  }

  function fmtAxisLine(prefix, v) {
    return prefix + ' ' + vectorHtml(v);
  }

  function vectorHtml(v) {
    return '(' +
      '<span class="axis-x">' + v.x + '</span>, ' +
      '<span class="axis-y">' + v.y + '</span>, ' +
      '<span class="axis-z">' + v.z + '</span>' +
      ')';
  }

  function sameBlock(a, b) {
    return Boolean(a && b) && a.x === b.x && a.y === b.y && a.z === b.z;
  }

  function sameThread(a, b) {
    if (!a && !b) {
      return true;
    }
    return Boolean(a && b) && a.x === b.x && a.y === b.y && a.z === b.z;
  }

  function shouldScopeThreadHover(launch) {
    return launch.threadsPerBlock > 1024 || launch.totalThreadsLaunched > 40000;
  }

  function getSelectedBlockEntry(host) {
    if (!state.selectedBlock || !host || !host.sceneData || !host.sceneData.blockMap) {
      return null;
    }
    return host.sceneData.blockMap[blockKey(state.selectedBlock)];
  }

  function renderStats(launch, derived) {
    var residency = derived.residency;
    var validation = derived.validation;
    var rows = [
      ["GPU", derived.gpu.name],
      ["Local viz tier", state.visualBudget.label],
      ["Local thread cap", getVisualizationThreadBudget().toLocaleString()],
      ["Grid size", fmt3(launch.grid)],
      ["Block size", fmt3(launch.block)],
      ["Total blocks launched", launch.totalBlocks.toLocaleString()],
      ["Threads per block", launch.threadsPerBlock.toLocaleString()],
      ["Warps per block", launch.warpsPerBlock.toLocaleString()],
      ["Resident blocks", residency.residentBlockCount.toLocaleString()],
      ["Queued blocks", residency.queuedBlockCount.toLocaleString()],
      ["Active blocks / SM", residency.occupancy.activeBlocksPerSM.toLocaleString()],
      ["Resident SMs", residency.residentSMCount.toLocaleString() + " / " + residency.smCount.toLocaleString()],
      ["Execution waves", residency.waveCount.toLocaleString()],
      ["Visualization mode", state.colorMode === "warp" ? "Warp" : "Thread"],
      ["Thread hover scope", shouldScopeThreadHover(launch)
        ? (state.selectedBlock ? "Selected block " + vectorHtml(state.selectedBlock) : "Click a block")
        : "Any resident block"],
      ["Launch validity", validation.isValid ? "Valid" : "Invalid"],
      ["Total threads launched", launch.totalThreadsLaunched.toLocaleString()],
      ["Selected block", state.selectedBlock ? vectorHtml(state.selectedBlock) : "-"],
      ["Hovered block", state.hovered && state.hovered.block ? "B(" + state.hovered.block.x + ", " + state.hovered.block.y + ", " + state.hovered.block.z + ")" : "-"]
    ];

    statsRoot.innerHTML = rows.map(function (row) {
      return '<div class="stats-row"><dt>' + row[0] + '</dt><dd>' + row[1] + '</dd></div>';
    }).join("");
  }

  function buildGpuUsagePanel() {
    var limits = clampKernelUsageToGpuLimits();
    var currentGpu = getCurrentGpuProfile();
    var options = Object.keys(math.GPU_PROFILES).map(function (key) {
      var selected = state.config.kernelUsage.gpuKey === key ? ' selected' : '';
      return '<option value="' + key + '"' + selected + '>' + math.GPU_PROFILES[key].name + '</option>';
    }).join("");

    gpuUsagePanelRoot.innerHTML =
      '<div class="gpu-grid">' +
        '<section class="gpu-card">' +
          '<h3>Kernel Inputs</h3>' +
          '<div class="gpu-field">' +
            '<label for="gpu-select">GPU Profile</label>' +
            '<select id="gpu-select">' + options + '</select>' +
          '</div>' +
          '<div class="gpu-field">' +
            '<label>Linked Block Size</label>' +
            '<div id="gpu-block-size" class="gpu-value"></div>' +
          '</div>' +
          '<div class="gpu-field">' +
            '<label for="registers-per-thread">Register Usage / Thread</label>' +
            '<input id="registers-per-thread" type="range" min="0" max="' + limits.maxRegistersPerThread + '" step="1" value="' + state.config.kernelUsage.registersPerThread + '">' +
            '<div id="registers-per-thread-value" class="gpu-value"></div>' +
          '</div>' +
          '<div class="gpu-field">' +
            '<label for="shared-mem-per-block">Shared Memory / Block (KB)</label>' +
            '<input id="shared-mem-per-block" type="range" min="0" max="' + limits.maxSharedMemoryPerBlockKB + '" step="1" value="' + state.config.kernelUsage.sharedMemoryPerBlockKB + '">' +
            '<div id="shared-mem-per-block-value" class="gpu-value"></div>' +
          '</div>' +
        '</section>' +
        '<section class="gpu-card">' +
          '<h3>Occupancy Estimate</h3>' +
          '<dl id="gpu-occupancy-list" class="gpu-list"></dl>' +
        '</section>' +
      '</div>';

    var gpuSelect = document.getElementById("gpu-select");
    var registersInput = document.getElementById("registers-per-thread");
    var sharedInput = document.getElementById("shared-mem-per-block");

    gpuSelect.addEventListener("change", function (event) {
      state.config.kernelUsage.gpuKey = event.currentTarget.value;
      buildGpuUsagePanel();
      redraw();
    });

    registersInput.addEventListener("input", function (event) {
      state.config.kernelUsage.registersPerThread = Number(event.currentTarget.value);
      redraw();
    });

    sharedInput.addEventListener("input", function (event) {
      state.config.kernelUsage.sharedMemoryPerBlockKB = Number(event.currentTarget.value);
      redraw();
    });

    renderGpuUsagePanel();
  }

  function renderGpuUsagePanel(derived) {
    var limits = clampKernelUsageToGpuLimits();
    var gpu = derived ? derived.gpu : getCurrentGpuProfile();
    var launch = math.computeLaunch(state.config);
    var occupancy = math.computeOccupancy(
      getKernelUsageForLaunch({ block: state.config.block }),
      gpu
    );
    var residency = derived ? derived.residency : getCurrentResidency(launch);
    var validation = derived ? derived.validation : getCurrentValidation(launch);

    document.getElementById("gpu-block-size").innerHTML = fmt3(state.config.block);
    document.getElementById("registers-per-thread-value").textContent =
      state.config.kernelUsage.registersPerThread + " regs/thread  |  max " + limits.maxRegistersPerThread + " for " + gpu.name;
    document.getElementById("shared-mem-per-block-value").textContent =
      state.config.kernelUsage.sharedMemoryPerBlockKB + " KB/block  |  max " + limits.maxSharedMemoryPerBlockKB + " KB on " + gpu.name;

    var rows = [
      ["GPU", gpu.name],
      ["Local viz budget", getVisualizationThreadBudget() + " (" + state.visualBudget.label + ")"],
      ["Streaming Multiprocessors", gpu.smCount],
      ["Max threads / block", gpu.maxThreadsPerBlock],
      ["Active blocks / SM", occupancy.activeBlocksPerSM],
      ["Active warps / SM", occupancy.activeWarpsPerSM + " / " + gpu.maxWarpsPerSM],
      ["Occupancy", Math.round(occupancy.occupancyRatio * 100) + "%"],
      ["Resident blocks / wave", residency.blocksPerWave],
      ["Queued blocks", residency.queuedBlockCount],
      ["Execution waves", residency.waveCount],
      ["Thread limiter", occupancy.limiters.maxThreadsPerSM],
      ["Warp limiter", occupancy.limiters.maxWarpsPerSM],
      ["Register limiter", occupancy.limiters.registersPerSM],
      ["Shared memory limiter", occupancy.limiters.sharedMemoryPerSM],
      ["Block limiter", occupancy.limiters.maxBlocksPerSM],
      ["Launch validity", validation.isValid ? "Valid" : validation.violations.map(function (v) { return v.code; }).join(", ")]
    ];

    document.getElementById("gpu-occupancy-list").innerHTML = rows.map(function (row) {
      return '<div class="gpu-list-row"><dt>' + row[0] + '</dt><dd>' + row[1] + '</dd></div>';
    }).join("");
  }

  function updateModeButtons() {
    threadModeButton.classList.toggle("active", state.colorMode === "thread");
    warpModeButton.classList.toggle("active", state.colorMode === "warp");
  }

  function buildControls() {
    normalizeBlockControlState();
    controlsRoot.innerHTML = controlSchema.map(function (group) {
      var fields = group.fields.map(function (field) {
        var key = field[0];
        var label = field[1];
        var min = field[2];
        var max = field[3];
        var id = group.prefix + "-" + key;
        var value = state.config[group.prefix][key];
        var inputHtml = "";
        if (group.inputType === "number") {
          inputHtml = '<input id="' + id + '" data-prefix="' + group.prefix + '" data-key="' + key + '" type="number" min="' + min + '" max="' + max + '" step="1" value="' + value + '">';
        } else if (group.inputType === "range-pow2") {
          inputHtml = '<input id="' + id + '" data-prefix="' + group.prefix + '" data-key="' + key + '" data-scale="pow2" type="range" min="0" max="' + math.powerOfTwoExponent(max, max) + '" step="1" value="' + math.powerOfTwoExponent(value, max) + '">';
        } else {
          inputHtml = '<input id="' + id + '" data-prefix="' + group.prefix + '" data-key="' + key + '" type="range" min="' + min + '" max="' + max + '" step="1" value="' + value + '">';
        }
        return (
          '<div class="field">' +
            '<label for="' + id + '">' + label + '</label>' +
            inputHtml +
            '<div class="field-value" data-value-for="' + id + '">' + value + '</div>' +
          '</div>'
        );
      }).join("");

      return (
        '<section class="control-group">' +
          '<h3>' + group.title + '</h3>' +
          '<div class="view-note">' +
            (group.prefix === "grid"
              ? "How many blocks to dispatch on X, Y, Z. Grid sliders go to " + math.MAX_GRID_DIMENSION + ", but local visualization is capped at " + getVisualizationThreadBudget() + " threads for this machine."
              : "How many threads each block contains on X, Y, Z. Block sliders move in powers of 2 up to (" + math.MAX_BLOCK_DIMENSION.x + ", " + math.MAX_BLOCK_DIMENSION.y + ", " + math.MAX_BLOCK_DIMENSION.z + "), but local visualization is capped at " + getVisualizationThreadBudget() + " threads for this machine.") +
          '</div>' +
          '<div class="control-grid">' + fields + '</div>' +
        '</section>'
      );
    }).join("");

    Array.prototype.forEach.call(controlsRoot.querySelectorAll("input"), function (input) {
      input.addEventListener("input", function (event) {
        var target = event.currentTarget;
        var proposedConfig = {
          grid: {
            x: state.config.grid.x,
            y: state.config.grid.y,
            z: state.config.grid.z
          },
          block: {
            x: state.config.block.x,
            y: state.config.block.y,
            z: state.config.block.z
          }
        };
        proposedConfig[target.dataset.prefix][target.dataset.key] = target.dataset.scale === "pow2"
          ? Math.pow(2, Number(target.value))
          : Number(target.value);

        if (math.computeLaunch(proposedConfig).totalThreadsLaunched > getVisualizationThreadBudget()) {
          target.value = target.dataset.scale === "pow2"
            ? String(math.powerOfTwoExponent(state.config[target.dataset.prefix][target.dataset.key], Number(target.max) ? Math.pow(2, Number(target.max)) : state.config[target.dataset.prefix][target.dataset.key]))
            : String(state.config[target.dataset.prefix][target.dataset.key]);
          combinedInfoRoot.textContent =
            'Launch capped at ' + getVisualizationThreadBudget() +
            ' threads for this local machine (' + state.visualBudget.label + ').';
          return;
        }

        var sanitizedConfig = math.sanitizeConfig(proposedConfig);
        state.config.grid = sanitizedConfig.grid;
        state.config.block = sanitizedConfig.block;
        lastValidConfig = {
          grid: { x: state.config.grid.x, y: state.config.grid.y, z: state.config.grid.z },
          block: { x: state.config.block.x, y: state.config.block.y, z: state.config.block.z }
        };
        var valueNode = controlsRoot.querySelector('[data-value-for="' + target.id + '"]');
        if (valueNode) {
          valueNode.textContent = state.config[target.dataset.prefix][target.dataset.key];
        }
        redraw();
      });
    });
  }

  function syncSelection(launch) {
    if (state.hovered && state.hovered.block) {
      state.hovered.block.x = Math.min(state.hovered.block.x, launch.grid.x - 1);
      state.hovered.block.y = Math.min(state.hovered.block.y, launch.grid.y - 1);
      state.hovered.block.z = Math.min(state.hovered.block.z, launch.grid.z - 1);
    }
    if (state.selectedBlock) {
      if (
        state.selectedBlock.x >= launch.grid.x ||
        state.selectedBlock.y >= launch.grid.y ||
        state.selectedBlock.z >= launch.grid.z
      ) {
        state.selectedBlock = null;
      }
    }
  }

  function createSceneHost(root) {
    var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    root.innerHTML = "";
    root.appendChild(renderer.domElement);
    renderer.domElement.classList.add("scene-canvas");

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(40, 1, 0.1, 1000);
    var ambient = new THREE.AmbientLight(0xffffff, 1.1);
    var key = new THREE.DirectionalLight(0xffffff, 1.35);
    key.position.set(8, 14, 12);
    var fill = new THREE.DirectionalLight(0xb8d9ff, 0.55);
    fill.position.set(-8, 5, -9);
    scene.add(ambient);
    scene.add(key);
    scene.add(fill);

    var rootGroup = new THREE.Group();
    scene.add(rootGroup);
    var overlayGroup = new THREE.Group();
    scene.add(overlayGroup);

    var raycaster = new THREE.Raycaster();
    var pointer = new THREE.Vector2();
    var orbit = {
      radius: 12,
      theta: Math.PI / 4,
      phi: 1.0,
      target: new THREE.Vector3(0, 0, 0),
      renderRadius: 12,
      renderTheta: Math.PI / 4,
      renderPhi: 1.0,
      renderTarget: new THREE.Vector3(0, 0, 0)
    };
    var dragState = {
      active: false,
      mode: null,
      lastX: 0,
      lastY: 0
    };
    var animationState = {
      lastFrameAt: performance.now(),
      rafId: 0,
      spawnAnimations: []
    };
    var thisHost = null;

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function updateCamera(deltaSeconds) {
      var easing = 1 - Math.exp(-Math.max(0.001, deltaSeconds) * 9.5);
      orbit.renderRadius = THREE.MathUtils.lerp(orbit.renderRadius, orbit.radius, easing);
      orbit.renderTheta = THREE.MathUtils.lerp(orbit.renderTheta, orbit.theta, easing);
      orbit.renderPhi = THREE.MathUtils.lerp(orbit.renderPhi, orbit.phi, easing);
      orbit.renderTarget.lerp(orbit.target, easing);

      var sinPhi = Math.sin(orbit.renderPhi);
      camera.position.set(
        orbit.renderTarget.x + orbit.renderRadius * sinPhi * Math.cos(orbit.renderTheta),
        orbit.renderTarget.y + orbit.renderRadius * Math.cos(orbit.renderPhi),
        orbit.renderTarget.z + orbit.renderRadius * sinPhi * Math.sin(orbit.renderTheta)
      );
      camera.lookAt(orbit.renderTarget);
    }

    function animateScene(timeSeconds) {
      var idleStrength = dragState.active ? 0.18 : (root.matches(":hover") ? 0.52 : 1);
      var driftY = Math.sin(timeSeconds * 0.86) * 0.07 * idleStrength;
      rootGroup.rotation.y = Math.sin(timeSeconds * 0.22) * 0.032 * idleStrength;
      rootGroup.rotation.x = Math.cos(timeSeconds * 0.17) * 0.012 * idleStrength;
      rootGroup.position.y = driftY;
      overlayGroup.position.y = driftY;

      key.position.x = 8 + Math.sin(timeSeconds * 0.62) * 1.15;
      key.position.y = 14 + Math.cos(timeSeconds * 0.48) * 0.55;
      key.intensity = 1.35 + Math.sin(timeSeconds * 1.28) * 0.08;
      fill.position.z = -9 + Math.cos(timeSeconds * 0.58) * 0.9;
      fill.intensity = 0.55 + Math.sin(timeSeconds * 0.94) * 0.06;
      ambient.intensity = 1.08 + Math.cos(timeSeconds * 0.44) * 0.04;

      if (thisHost && thisHost.hoverArtifacts) {
        if (thisHost.hoverArtifacts.selectedHighlight) {
          thisHost.hoverArtifacts.selectedHighlight.scale.setScalar(
            thisHost.hoverArtifacts.selectedBaseScale * (1 + Math.sin(timeSeconds * 2.2) * 0.018)
          );
        }
        if (thisHost.hoverArtifacts.shellHighlight) {
          thisHost.hoverArtifacts.shellHighlight.scale.setScalar(
            thisHost.hoverArtifacts.baseShellScale * (1 + Math.sin(timeSeconds * 5.2) * 0.022)
          );
          thisHost.hoverArtifacts.shellHighlight.material.opacity =
            thisHost.hoverArtifacts.baseShellOpacity + Math.sin(timeSeconds * 5.2) * 0.028;
        }
        if (thisHost.hoverArtifacts.shellWire) {
          thisHost.hoverArtifacts.shellWire.scale.setScalar(
            thisHost.hoverArtifacts.baseWireScale * (1 + Math.cos(timeSeconds * 6.1) * 0.02)
          );
          thisHost.hoverArtifacts.shellWire.material.opacity =
            thisHost.hoverArtifacts.baseWireOpacity + Math.sin(timeSeconds * 6.1) * 0.06;
        }
        if (thisHost.hoverArtifacts.threadHighlight) {
          thisHost.hoverArtifacts.threadHighlight.scale.setScalar(
            thisHost.hoverArtifacts.baseThreadScale * (1 + Math.sin(timeSeconds * 8.4) * 0.042)
          );
          thisHost.hoverArtifacts.threadHighlight.material.emissiveIntensity =
            0.42 + Math.sin(timeSeconds * 8.4) * 0.08;
        }
        if (thisHost.hoverArtifacts.label) {
          thisHost.hoverArtifacts.label.position.y =
            thisHost.hoverArtifacts.baseLabelY + Math.sin(timeSeconds * 3.6) * 0.08;
        }
      }
    }

    function registerSpawn(node, options) {
      if (!node) {
        return node;
      }
      var settings = options || {};
      var baseScale = node.scale.clone();
      var startScale = typeof settings.startScale === "number" ? settings.startScale : 0.94;
      var lift = typeof settings.lift === "number" ? settings.lift : 0.18;
      var duration = typeof settings.duration === "number" ? settings.duration : 150;
      var materials = [];
      if (node.material) {
        materials = Array.isArray(node.material) ? node.material.slice() : [node.material];
      }
      node.scale.set(baseScale.x * startScale, baseScale.y * startScale, baseScale.z * startScale);
      node.position.y += lift;
      materials.forEach(function (material) {
        if (material && typeof material.opacity === "number") {
          material.transparent = true;
          material.opacity *= 0.18;
        }
      });
      animationState.spawnAnimations.push({
        node: node,
        startAt: performance.now(),
        duration: duration,
        lift: lift,
        baseScale: baseScale,
        baseY: node.position.y - lift,
        materials: materials.map(function (material) {
          return {
            material: material,
            targetOpacity: material && typeof material.opacity === "number" ? material.opacity / 0.18 : null
          };
        })
      });
      return node;
    }

    function animateSpawnAnimations(frameAt) {
      var index = animationState.spawnAnimations.length - 1;
      for (; index >= 0; index -= 1) {
        var entry = animationState.spawnAnimations[index];
        if (!entry.node || !entry.node.parent) {
          animationState.spawnAnimations.splice(index, 1);
          continue;
        }
        var progress = Math.min(1, (frameAt - entry.startAt) / entry.duration);
        var eased = 1 - Math.pow(1 - progress, 3);
        var popStrength = Math.sin(progress * Math.PI) * 0.12;
        var scaleFactor = 0.94 + (0.06 * eased) + popStrength;
        entry.node.scale.set(
          entry.baseScale.x * scaleFactor,
          entry.baseScale.y * scaleFactor,
          entry.baseScale.z * scaleFactor
        );
        entry.node.position.y = entry.baseY + ((1 - eased) * entry.lift);
        entry.materials.forEach(function (materialEntry) {
          if (materialEntry.material && materialEntry.targetOpacity !== null) {
            materialEntry.material.opacity = materialEntry.targetOpacity * (0.18 + (0.82 * eased));
          }
        });
        if (progress >= 1) {
          entry.node.scale.copy(entry.baseScale);
          entry.node.position.y = entry.baseY;
          entry.materials.forEach(function (materialEntry) {
            if (materialEntry.material && materialEntry.targetOpacity !== null) {
              materialEntry.material.opacity = materialEntry.targetOpacity;
            }
          });
          animationState.spawnAnimations.splice(index, 1);
        }
      }
    }

    function resize() {
      var width = root.clientWidth || 920;
      var height = root.clientHeight || 520;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }

    function syncInteractionState() {
      root.classList.toggle("is-dragging", dragState.active);
      root.classList.toggle("is-hovered", !dragState.active && root.matches(":hover"));
    }

    function render(nowMs) {
      var startedAt = performance.now();
      var frameAt = typeof nowMs === "number" ? nowMs : startedAt;
      var deltaSeconds = Math.min((frameAt - animationState.lastFrameAt) / 1000, 0.05);
      if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) {
        deltaSeconds = 1 / 60;
      }
      animationState.lastFrameAt = frameAt;
      updateCamera(deltaSeconds);
      animateSpawnAnimations(frameAt);
      animateScene(frameAt / 1000);
      renderer.render(scene, camera);
      trackRenderPerformance(performance.now() - startedAt);
    }

    function tick(nowMs) {
      render(nowMs);
      animationState.rafId = window.requestAnimationFrame(tick);
    }

    renderer.domElement.addEventListener("contextmenu", function (event) {
      event.preventDefault();
    });

    renderer.domElement.addEventListener("mouseenter", function () {
      root.classList.toggle("is-hovered", !dragState.active);
    });

    renderer.domElement.addEventListener("mouseleave", function () {
      root.classList.remove("is-hovered");
    });

    renderer.domElement.addEventListener("mousedown", function (event) {
      dragState.active = true;
      dragState.mode = event.button === 2 ? "pan" : "orbit";
      dragState.lastX = event.clientX;
      dragState.lastY = event.clientY;
      syncInteractionState();
    });

    window.addEventListener("mouseup", function () {
      dragState.active = false;
      dragState.mode = null;
      syncInteractionState();
    });

    window.addEventListener("mousemove", function (event) {
      if (!dragState.active) {
        return;
      }

      var dx = event.clientX - dragState.lastX;
      var dy = event.clientY - dragState.lastY;
      dragState.lastX = event.clientX;
      dragState.lastY = event.clientY;

      if (dragState.mode === "orbit") {
        orbit.theta -= dx * 0.01;
        orbit.phi = clamp(orbit.phi + dy * 0.01, 0.2, Math.PI - 0.2);
      } else if (dragState.mode === "pan") {
        var panScale = orbit.radius * 0.0018;
        var forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        var right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
        var up = new THREE.Vector3().copy(camera.up).normalize();
        orbit.target.addScaledVector(right, -dx * panScale);
        orbit.target.addScaledVector(up, dy * panScale);
      }

      syncInteractionState();
      render();
    });

    renderer.domElement.addEventListener("wheel", function (event) {
      event.preventDefault();
      orbit.radius = clamp(orbit.radius * (1 + Math.sign(event.deltaY) * 0.08), 2.5, 220);
      syncInteractionState();
      render();
    }, { passive: false });

    resize();
    render();
    animationState.rafId = window.requestAnimationFrame(tick);
    window.addEventListener("resize", function () {
      resize();
      render();
    });

    var resizeObserver = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(function () {
        resize();
        render();
      });
      resizeObserver.observe(combinedViewportShellRoot || root.parentElement || root);
    }

    thisHost = {
      scene: scene,
      camera: camera,
      renderer: renderer,
      rootGroup: rootGroup,
      overlayGroup: overlayGroup,
      raycaster: raycaster,
      pointer: pointer,
      orbit: orbit,
      hoverArtifacts: null,
      pickTargets: [],
      registerSpawn: registerSpawn,
      lastRay: new THREE.Ray(),
      lastPointerPixels: { x: 0, y: 0, width: 1, height: 1 },
      resize: resize,
      render: render,
      resetView: function () {
        orbit.radius = 12;
        orbit.theta = Math.PI / 4;
        orbit.phi = 1.0;
        orbit.target.set(0, 0, 0);
        render();
      },
      isDragging: function () {
        return dragState.active;
      },
      pick: function (event, onObject) {
        var rect = renderer.domElement.getBoundingClientRect();
        this.lastPointerPixels.x = event.clientX - rect.left;
        this.lastPointerPixels.y = event.clientY - rect.top;
        this.lastPointerPixels.width = rect.width;
        this.lastPointerPixels.height = rect.height;
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        this.lastRay.copy(raycaster.ray);
        var targets = this.pickTargets && this.pickTargets.length > 0 ? this.pickTargets : rootGroup.children;
        var hits = raycaster.intersectObjects(targets, true);
        if (hits.length > 0) {
          var preferred = null;
          var i;
          for (i = 0; i < hits.length; i += 1) {
            if (
              hits[i].object &&
              hits[i].object.userData &&
              typeof hits[i].object.userData.pickPriority === "number"
            ) {
              if (
                !preferred ||
                hits[i].object.userData.pickPriority > preferred.object.userData.pickPriority
              ) {
                preferred = hits[i];
              }
            }
          }
          onObject(preferred || hits[0]);
          return;
        }
        onObject(null);
      }
    };

    return thisHost;
  }

  function addSpawnedObject(host, group, node, options) {
    group.add(node);
    if (host && host.registerSpawn) {
      host.registerSpawn(node, options);
    }
    return node;
  }

  function disposeObject(node) {
    if (!node) {
      return;
    }
    if (node.geometry) {
      node.geometry.dispose();
    }
    if (node.material) {
      if (Array.isArray(node.material)) {
        node.material.forEach(function (material) { material.dispose(); });
      } else {
        node.material.dispose();
      }
    }
  }

  function clearGroup(group) {
    while (group.children.length > 0) {
      var child = group.children[0];
      group.remove(child);
      disposeObject(child);
    }
  }

  function centerOffset(dims, spacing) {
    return new THREE.Vector3(
      -((dims.x - 1) * spacing) / 2,
      -((dims.y - 1) * spacing) / 2,
      -((dims.z - 1) * spacing) / 2
    );
  }

  function fitCamera(host, dims, spacing, factor) {
    var extent = Math.max(dims.x, dims.y, dims.z) * spacing;
    host.orbit.radius = extent * (factor || 1.5);
    host.orbit.theta = Math.PI / 4;
    host.orbit.phi = 1.0;
    host.orbit.target.set(0, 0, 0);
    host.render();
  }

  function blockMaterial(hovered) {
    return new THREE.MeshStandardMaterial({
      color: hovered ? 0x4f93f0 : 0x7faf93,
      transparent: true,
      opacity: hovered ? 0.26 : 0.14,
      roughness: 0.5,
      metalness: 0.02,
      depthWrite: false
    });
  }

  function threadColor(localX, localY, localZ, dims, hovered) {
    if (state.colorMode === "thread") {
      var nx = dims.x <= 1 ? 0 : localX / (dims.x - 1);
      var ny = dims.y <= 1 ? 0 : localY / (dims.y - 1);
      var nz = dims.z <= 1 ? 0 : localZ / (dims.z - 1);
      var baseHue = (0.03 + nx * 0.2 + ny * 0.12 + nz * 0.08) % 1;
      var threadModeColor = new THREE.Color();
      threadModeColor.setHSL(baseHue, hovered ? 0.82 : 0.74, hovered ? 0.58 : 0.5);
      return threadModeColor;
    }

    var warpId = math.getWarpId({ x: localX, y: localY, z: localZ }, dims);
    var hue = ((warpId * 0.17) % 1);
    var saturation = hovered ? 0.82 : 0.76;
    var lightness = hovered ? 0.58 : 0.5;
    var color = new THREE.Color();
    color.setHSL(hue, saturation, lightness);
    return color;
  }

  function makeSpriteLabel(text, scaleX, scaleY, options) {
    options = options || {};
    var canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 220;
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var baseX = 48;

    ctx.fillStyle = options.kickerColor || "#2f5f49";
    ctx.font = "600 40px Bahnschrift";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.lineJoin = "round";
    ctx.lineWidth = 12;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.96)";
    ctx.shadowColor = "rgba(255, 255, 255, 0.98)";
    ctx.shadowBlur = 18;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    if (options.kicker) {
      ctx.strokeText(options.kicker, baseX, 68);
      ctx.fillText(options.kicker, baseX, 68);
    }

    if (options.vectorParts && options.vectorParts.length) {
      drawVectorText(ctx, options.vectorParts, baseX, 154, options.textColor || "#183024");
    } else {
      ctx.fillStyle = options.textColor || "#183024";
      ctx.font = "600 64px Bahnschrift";
      ctx.strokeText(text, baseX, 154);
      ctx.fillText(text, baseX, 154);
    }
    var texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    var material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    var sprite = new THREE.Sprite(material);
    sprite.scale.set(scaleX || 5.8, scaleY || 1.34, 1);
    return sprite;
  }

  function drawVectorText(ctx, parts, startX, baselineY, fallbackColor) {
    var x = startX;
    ctx.font = "600 64px Bahnschrift";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";

    parts.forEach(function (part) {
      ctx.fillStyle = part.color || fallbackColor;
      ctx.strokeText(part.text, x, baselineY);
      ctx.fillText(part.text, x, baselineY);
      x += ctx.measureText(part.text).width;
    });
  }

  function vectorSpriteParts(prefix, v) {
    return [
      { text: prefix + " (", color: "#183024" },
      { text: String(v.x), color: "#d14b45" },
      { text: ", ", color: "#183024" },
      { text: String(v.y), color: "#c9a227" },
      { text: ", ", color: "#183024" },
      { text: String(v.z), color: "#356fd6" },
      { text: ")", color: "#183024" }
    ];
  }

  function axisSummaryParts(label, blockValue, threadValue) {
    return [
      { text: label + "  ", color: "#183024" },
      { text: "Blocks ", color: "#183024" },
      { text: String(blockValue), color: "#183024" },
      { text: "  Threads ", color: "#183024" },
      { text: String(threadValue), color: "#183024" }
    ];
  }

  function blockKey(block) {
    return block.x + ":" + block.y + ":" + block.z;
  }

  function computePackedDims(count) {
    var x = Math.max(1, Math.ceil(Math.pow(count, 1 / 3)));
    var y = Math.max(1, Math.ceil(Math.sqrt(count / x)));
    var z = Math.max(1, Math.ceil(count / (x * y)));
    return { x: x, y: y, z: z };
  }

  function fitCameraToScene(host, factor) {
    if (!host.sceneData) {
      return;
    }
    host.orbit.radius = Math.max(6, host.sceneData.boundingRadius * (factor || 2.15));
    host.orbit.theta = Math.PI / 4;
    host.orbit.phi = 1.0;
    host.orbit.target.set(0, 0, 0);
    host.render();
  }

  function getSceneLayoutSignature() {
    return [
      state.config.grid.x,
      state.config.grid.y,
      state.config.grid.z,
      state.config.block.x,
      state.config.block.y,
      state.config.block.z,
      state.config.kernelUsage.gpuKey,
      state.config.kernelUsage.registersPerThread,
      state.config.kernelUsage.sharedMemoryPerBlockKB,
      state.viewOptions.organizeBySm ? 1 : 0,
      state.viewOptions.showSmMemory ? 1 : 0
    ].join(":");
  }

  function trackRenderPerformance(renderMs) {
    state.performance.recentRenderMs.push(renderMs);
    if (state.performance.recentRenderMs.length > 5) {
      state.performance.recentRenderMs.shift();
    }
    var average = state.performance.recentRenderMs.reduce(function (sum, value) {
      return sum + value;
    }, 0) / state.performance.recentRenderMs.length;
    var adaptive = math.computeAdaptiveVisualizationBudget({
      currentMaxThreads: getVisualizationThreadBudget(),
      baselineMaxThreads: state.visualBudget.baselineMaxThreads,
      averageFrameMs: average
    });

    if (adaptive.shouldReduce) {
      state.performance.slowRenderCount += 1;
      if (
        state.performance.slowRenderCount >= 3 &&
        adaptive.nextMaxThreads < state.visualBudget.maxThreads &&
        !state.performance.reducingBudget
      ) {
        state.performance.reducingBudget = true;
        state.visualBudget.maxThreads = adaptive.nextMaxThreads;
        state.performance.slowRenderCount = 0;
        buildControls();
        redraw();
        state.performance.reducingBudget = false;
      }
      return;
    }

    state.performance.slowRenderCount = 0;
  }

  function buildFlatSceneData(launch, derived) {
    var residency = derived.residency;
    var threadUnit = 0.16;
    var threadGap = 0.08;
    var threadSpacing = threadUnit + threadGap;
    var shellInnerX = Math.max(0.4, launch.block.x * threadSpacing - threadGap);
    var shellInnerY = Math.max(0.4, launch.block.y * threadSpacing - threadGap);
    var shellInnerZ = Math.max(0.4, launch.block.z * threadSpacing - threadGap);
    var shellPadding = 0.22;
    var shellSizeX = shellInnerX + shellPadding;
    var shellSizeY = shellInnerY + shellPadding;
    var shellSizeZ = shellInnerZ + shellPadding;
    var blockGap = 0.62;
    var blockPitchX = shellSizeX + blockGap;
    var blockPitchY = shellSizeY + blockGap;
    var blockPitchZ = shellSizeZ + blockGap;
    var offset = new THREE.Vector3(
      -((launch.grid.x - 1) * blockPitchX) / 2,
      -((launch.grid.y - 1) * blockPitchY) / 2,
      -((launch.grid.z - 1) * blockPitchZ) / 2
    );
    var blockLocalOffset = new THREE.Vector3(
      -((launch.block.x - 1) * threadSpacing) / 2,
      -((launch.block.y - 1) * threadSpacing) / 2,
      -((launch.block.z - 1) * threadSpacing) / 2
    );
    var shellEntries = [];
    var threadEntries = [];
    var blockMap = {};
    var allBlocks = math.buildBlockList(launch, launch.totalBlocks);

    allBlocks.forEach(function (blockEntry, index) {
      var bx = blockEntry.block.x;
      var by = blockEntry.block.y;
      var bz = blockEntry.block.z;
      var blockCenter = new THREE.Vector3(
        offset.x + bx * blockPitchX,
        offset.y + by * blockPitchY,
        offset.z + bz * blockPitchZ
      );
      var status = index < residency.residentBlockCount ? "resident" : "queued";
      var shellEntry = {
        block: { x: bx, y: by, z: bz },
        sm: null,
        center: blockCenter.clone(),
        status: status
      };
      shellEntries.push(shellEntry);
      blockMap[blockKey(shellEntry.block)] = shellEntry;

      if (status !== "resident") {
        return;
      }

      for (var tz = 0; tz < launch.block.z; tz += 1) {
        for (var ty = 0; ty < launch.block.y; ty += 1) {
          for (var tx = 0; tx < launch.block.x; tx += 1) {
            threadEntries.push({
              block: shellEntry.block,
              sm: null,
              status: "resident",
              thread: { x: tx, y: ty, z: tz },
              warpId: math.getWarpId({ x: tx, y: ty, z: tz }, launch.block),
              position: new THREE.Vector3(
                blockCenter.x + blockLocalOffset.x + tx * threadSpacing,
                blockCenter.y + blockLocalOffset.y + ty * threadSpacing,
                blockCenter.z + blockLocalOffset.z + tz * threadSpacing
              )
            });
          }
        }
      }
    });

    var totalSizeX = shellSizeX + Math.max(0, launch.grid.x - 1) * blockPitchX;
    var totalSizeY = shellSizeY + Math.max(0, launch.grid.y - 1) * blockPitchY;
    var totalSizeZ = shellSizeZ + Math.max(0, launch.grid.z - 1) * blockPitchZ;
    var boundingRadius = Math.sqrt(totalSizeX * totalSizeX + totalSizeY * totalSizeY + totalSizeZ * totalSizeZ) * 0.5 + 0.86;

    return {
      residency: residency,
      shellSizeX: shellSizeX,
      shellSizeY: shellSizeY,
      shellSizeZ: shellSizeZ,
      smSizeX: 0,
      smSizeY: 0,
      smSizeZ: 0,
      smLabelLift: 0.86,
      threadUnit: threadUnit,
      threadSpacing: threadSpacing,
      sceneSpacing: Math.max(blockPitchX, blockPitchY, blockPitchZ),
      totalSizeX: totalSizeX,
      totalSizeY: totalSizeY,
      totalSizeZ: totalSizeZ,
      boundingRadius: boundingRadius,
      smEntries: [],
      queueSmEntries: [],
      shellEntries: shellEntries.filter(function (entry) { return entry.status === "resident"; }),
      queueShellEntries: shellEntries.filter(function (entry) { return entry.status === "queued"; }),
      threadEntries: threadEntries,
      blockMap: blockMap
    };
  }

  function buildSceneData(launch, derived) {
    if (!state.viewOptions.organizeBySm) {
      return buildFlatSceneData(launch, derived);
    }
    var residency = derived.residency;
    var threadUnit = 0.16;
    var threadGap = 0.08;
    var threadSpacing = threadUnit + threadGap;
    var shellInnerX = Math.max(0.4, launch.block.x * threadSpacing - threadGap);
    var shellInnerY = Math.max(0.4, launch.block.y * threadSpacing - threadGap);
    var shellInnerZ = Math.max(0.4, launch.block.z * threadSpacing - threadGap);
    var shellPadding = 0.22;
    var shellSizeX = shellInnerX + shellPadding;
    var shellSizeY = shellInnerY + shellPadding;
    var shellSizeZ = shellInnerZ + shellPadding;
    var blockGap = 0.62;
    var blockPitchX = shellSizeX + blockGap;
    var blockPitchY = shellSizeY + blockGap;
    var blockPitchZ = shellSizeZ + blockGap;
    var blockLocalOffset = new THREE.Vector3(
      -((launch.block.x - 1) * threadSpacing) / 2,
      -((launch.block.y - 1) * threadSpacing) / 2,
      -((launch.block.z - 1) * threadSpacing) / 2
    );
    var smPadding = 0.52;
    var smSpacing = 1.55;
    var smLabelLift = 0.86;
    var shellEntries = [];
    var threadEntries = [];
    var blockMap = {};
    var smEntries = [];
    var queueSmEntries = [];
    var queueShellEntries = [];
    var activeBlocksPerSM = Math.max(1, residency.occupancy.activeBlocksPerSM);
    var smBlockDims = computePackedDims(Math.max(1, activeBlocksPerSM));
    var smSizeX = smBlockDims.x * shellSizeX + Math.max(0, smBlockDims.x - 1) * blockGap + smPadding * 2;
    var smSizeY = smBlockDims.y * shellSizeY + Math.max(0, smBlockDims.y - 1) * blockGap + smPadding * 2;
    var smSizeZ = smBlockDims.z * shellSizeZ + Math.max(0, smBlockDims.z - 1) * blockGap + smPadding * 2;
    var smPitchX = smSizeX + smSpacing;
    var smPitchY = smSizeY + smSpacing;
    var smPitchZ = smSizeZ + smSpacing;
    var smGridDims = computePackedDims(Math.max(1, residency.residentSMCount));
    var smOffset = new THREE.Vector3(
      -((smGridDims.x - 1) * smPitchX) / 2,
      -((smGridDims.y - 1) * smPitchY) / 2,
      -((smGridDims.z - 1) * smPitchZ) / 2
    );
    var gpuProfile = getCurrentGpuProfile();
    var allBlocks = math.buildBlockList(launch, launch.totalBlocks);
    var residentBlocks = allBlocks.slice(0, residency.residentBlockCount);
    var queuedBlocks = allBlocks.slice(residency.residentBlockCount);
    var smIndex;

    for (smIndex = 0; smIndex < residency.residentSMCount; smIndex += 1) {
      var smBlocks = residentBlocks.slice(
        smIndex * activeBlocksPerSM,
        (smIndex + 1) * activeBlocksPerSM
      );
      if (smBlocks.length === 0) {
        continue;
      }
      var smCoord = {
        x: smIndex % smGridDims.x,
        y: Math.floor(smIndex / smGridDims.x) % smGridDims.y,
        z: Math.floor(smIndex / (smGridDims.x * smGridDims.y))
      };
      var smCenter = new THREE.Vector3(
        smOffset.x + smCoord.x * smPitchX,
        smOffset.y + smCoord.y * smPitchY,
        smOffset.z + smCoord.z * smPitchZ
      );
      var smBlockOffset = new THREE.Vector3(
        -((smBlockDims.x - 1) * blockPitchX) / 2,
        -((smBlockDims.y - 1) * blockPitchY) / 2,
        -((smBlockDims.z - 1) * blockPitchZ) / 2
      );
      var smEntry = {
        sm: smIndex,
        center: smCenter.clone(),
        size: { x: smSizeX, y: smSizeY, z: smSizeZ },
        sharedMemoryKB: smBlocks.length * residency.occupancy.sharedMemoryPerBlockKB,
        sharedMemoryRatio: gpuProfile.sharedMemoryPerSMKB > 0
          ? Math.min(1, (smBlocks.length * residency.occupancy.sharedMemoryPerBlockKB) / gpuProfile.sharedMemoryPerSMKB)
          : 0,
        blocks: []
      };
      smEntries.push(smEntry);

      smBlocks.forEach(function (blockEntry, localIndex) {
        var localCoord = {
          x: localIndex % smBlockDims.x,
          y: Math.floor(localIndex / smBlockDims.x) % smBlockDims.y,
          z: Math.floor(localIndex / (smBlockDims.x * smBlockDims.y))
        };
        var blockCenter = new THREE.Vector3(
          smCenter.x + smBlockOffset.x + localCoord.x * blockPitchX,
          smCenter.y + smBlockOffset.y + localCoord.y * blockPitchY,
          smCenter.z + smBlockOffset.z + localCoord.z * blockPitchZ
        );
        var block = {
          x: blockEntry.block.x,
          y: blockEntry.block.y,
          z: blockEntry.block.z
        };
        var shellEntry = {
          block: block,
          sm: smIndex,
          center: blockCenter.clone(),
          status: "resident"
        };
        shellEntries.push(shellEntry);
        smEntry.blocks.push(shellEntry);
        blockMap[blockKey(block)] = shellEntry;

        for (var tz = 0; tz < launch.block.z; tz += 1) {
          for (var ty = 0; ty < launch.block.y; ty += 1) {
            for (var tx = 0; tx < launch.block.x; tx += 1) {
                threadEntries.push({
                  block: block,
                  sm: smIndex,
                  status: "resident",
                  thread: { x: tx, y: ty, z: tz },
                  warpId: math.getWarpId({ x: tx, y: ty, z: tz }, launch.block),
                  position: new THREE.Vector3(
                  blockCenter.x + blockLocalOffset.x + tx * threadSpacing,
                  blockCenter.y + blockLocalOffset.y + ty * threadSpacing,
                  blockCenter.z + blockLocalOffset.z + tz * threadSpacing
                )
              });
            }
          }
        }
      });
    }

    if (queuedBlocks.length > 0) {
      var queuedSmCount = Math.ceil(queuedBlocks.length / activeBlocksPerSM);
      var queueSmGridDims = computePackedDims(Math.max(1, queuedSmCount));
      var residentRightEdge = smEntries.reduce(function (maxX, entry) {
        return Math.max(maxX, entry.center.x + entry.size.x * 0.5);
      }, -Infinity);
      if (!Number.isFinite(residentRightEdge)) {
        residentRightEdge = smSizeX * 0.5;
      }
      var queueClearance = smSizeX + smSpacing * 2;
      var queueSmOffset = new THREE.Vector3(
        residentRightEdge + queueClearance + smSizeX * 0.5,
        -((queueSmGridDims.y - 1) * smPitchY) / 2,
        -((queueSmGridDims.z - 1) * smPitchZ) / 2
      );

      for (smIndex = 0; smIndex < queuedSmCount; smIndex += 1) {
        var queuedSmBlocks = queuedBlocks.slice(
          smIndex * activeBlocksPerSM,
          (smIndex + 1) * activeBlocksPerSM
        );
        var queuedSmCoord = {
          x: smIndex % queueSmGridDims.x,
          y: Math.floor(smIndex / queueSmGridDims.x) % queueSmGridDims.y,
          z: Math.floor(smIndex / (queueSmGridDims.x * queueSmGridDims.y))
        };
        var queuedSmCenter = new THREE.Vector3(
          queueSmOffset.x + queuedSmCoord.x * smPitchX,
          queueSmOffset.y + queuedSmCoord.y * smPitchY,
          queueSmOffset.z + queuedSmCoord.z * smPitchZ
        );
        var queuedSmEntry = {
          sm: residency.residentSMCount + smIndex,
          center: queuedSmCenter.clone(),
          size: { x: smSizeX, y: smSizeY, z: smSizeZ },
          sharedMemoryKB: queuedSmBlocks.length * residency.occupancy.sharedMemoryPerBlockKB,
          status: "queued",
          blocks: []
        };
        queueSmEntries.push(queuedSmEntry);

        var queuedSmBlockOffset = new THREE.Vector3(
          -((smBlockDims.x - 1) * blockPitchX) / 2,
          -((smBlockDims.y - 1) * blockPitchY) / 2,
          -((smBlockDims.z - 1) * blockPitchZ) / 2
        );

        queuedSmBlocks.forEach(function (blockEntry, localIndex) {
          var localCoord = {
            x: localIndex % smBlockDims.x,
            y: Math.floor(localIndex / smBlockDims.x) % smBlockDims.y,
            z: Math.floor(localIndex / (smBlockDims.x * smBlockDims.y))
          };
          var center = new THREE.Vector3(
            queuedSmCenter.x + queuedSmBlockOffset.x + localCoord.x * blockPitchX,
            queuedSmCenter.y + queuedSmBlockOffset.y + localCoord.y * blockPitchY,
            queuedSmCenter.z + queuedSmBlockOffset.z + localCoord.z * blockPitchZ
          );
          var queuedEntry = {
            block: {
              x: blockEntry.block.x,
              y: blockEntry.block.y,
              z: blockEntry.block.z
            },
            sm: null,
            center: center,
            status: "queued"
          };
          queuedSmEntry.blocks.push(queuedEntry);
          queueShellEntries.push(queuedEntry);
          blockMap[blockKey(queuedEntry.block)] = queuedEntry;
        });
      }
    }

    var totalResidentX = smEntries.length > 0
      ? smEntries.reduce(function (maxX, entry) {
          return Math.max(maxX, entry.center.x + entry.size.x * 0.5);
        }, -Infinity) - smEntries.reduce(function (minX, entry) {
          return Math.min(minX, entry.center.x - entry.size.x * 0.5);
        }, Infinity)
      : smSizeX;
    var queueWidth = queuedBlocks.length > 0
      ? (smSizeX + Math.max(0, computePackedDims(Math.ceil(queuedBlocks.length / activeBlocksPerSM)).x - 1) * smPitchX)
      : 0;
    var totalSizeX = totalResidentX + (queuedBlocks.length > 0 ? queueWidth + smSizeX + smSpacing * 3 : 0);
    var totalSizeY = Math.max(
      smSizeY + Math.max(0, smGridDims.y - 1) * smPitchY,
      queuedBlocks.length > 0 ? smSizeY + Math.max(0, computePackedDims(Math.ceil(queuedBlocks.length / activeBlocksPerSM)).y - 1) * smPitchY : 0
    );
    var totalSizeZ = Math.max(
      smSizeZ + Math.max(0, smGridDims.z - 1) * smPitchZ,
      queuedBlocks.length > 0 ? smSizeZ + Math.max(0, computePackedDims(Math.ceil(queuedBlocks.length / activeBlocksPerSM)).z - 1) * smPitchZ : 0
    );
    var boundingRadius = Math.sqrt(totalSizeX * totalSizeX + totalSizeY * totalSizeY + totalSizeZ * totalSizeZ) * 0.5 + smLabelLift;

    return {
      residency: residency,
      shellSizeX: shellSizeX,
      shellSizeY: shellSizeY,
      shellSizeZ: shellSizeZ,
      smSizeX: smSizeX,
      smSizeY: smSizeY,
      smSizeZ: smSizeZ,
      smLabelLift: smLabelLift,
      threadUnit: threadUnit,
      threadSpacing: threadSpacing,
      sceneSpacing: Math.max(smPitchX, smPitchY, smPitchZ),
      totalSizeX: totalSizeX,
      totalSizeY: totalSizeY,
      totalSizeZ: totalSizeZ,
      boundingRadius: boundingRadius,
      smEntries: smEntries,
      queueSmEntries: queueSmEntries,
      shellEntries: shellEntries,
      queueShellEntries: queueShellEntries,
      threadEntries: threadEntries,
      blockMap: blockMap
    };
  }

  function addDimensionLabels(host, launch) {
    if (!host.sceneData) {
      return;
    }

    var data = host.sceneData;
    var topY = data.totalSizeY * 0.58 + 0.9;
    var leftX = -data.totalSizeX * 0.62 - 0.8;
    var frontZ = data.totalSizeZ * 0.62 + 0.8;

    var xLabel = makeSpriteLabel(
      "",
      4.5,
      0.82,
      {
        kicker: "X Axis",
        fill: "rgba(251, 241, 240, 0.96)",
        stroke: "rgba(209, 75, 69, 0.20)",
        kickerColor: "#b53f39",
        textColor: "#5f1f1b",
        vectorParts: axisSummaryParts("Span", launch.grid.x, launch.block.x)
      }
    );
    xLabel.position.set(0, topY, -data.totalSizeZ * 0.56);
    addSpawnedObject(host, host.overlayGroup, xLabel, { startScale: 0.9, lift: 0.12, duration: 135 });

    var yLabel = makeSpriteLabel(
      "",
      4.5,
      0.82,
      {
        kicker: "Y Axis",
        fill: "rgba(251, 248, 238, 0.96)",
        stroke: "rgba(201, 162, 39, 0.24)",
        kickerColor: "#a88418",
        textColor: "#5b4710",
        vectorParts: axisSummaryParts("Span", launch.grid.y, launch.block.y)
      }
    );
    yLabel.position.set(leftX, 0, -data.totalSizeZ * 0.56);
    addSpawnedObject(host, host.overlayGroup, yLabel, { startScale: 0.9, lift: 0.12, duration: 135 });

    var zLabel = makeSpriteLabel(
      "",
      4.5,
      0.82,
      {
        kicker: "Z Axis",
        fill: "rgba(240, 245, 252, 0.96)",
        stroke: "rgba(53, 111, 214, 0.22)",
        kickerColor: "#2f63bc",
        textColor: "#1d3567",
        vectorParts: axisSummaryParts("Span", launch.grid.z, launch.block.z)
      }
    );
    zLabel.position.set(data.totalSizeX * 0.56, -data.totalSizeY * 0.58, frontZ);
    addSpawnedObject(host, host.overlayGroup, zLabel, { startScale: 0.9, lift: 0.12, duration: 135 });
  }

  function addSmLabels(host) {
    if (!host.sceneData) {
      return;
    }

    host.sceneData.smEntries.forEach(function (entry) {
      var smLabel = makeSpriteLabel(
        "",
        4.8,
        0.8,
        {
          kicker: "Streaming Multiprocessor",
          fill: "rgba(244, 248, 255, 0.95)",
          stroke: "rgba(77, 110, 184, 0.18)",
          kickerColor: "#4669b3",
          vectorParts: [
            { text: "SM " + entry.sm + "  ", color: "#183024" },
            { text: "Blocks ", color: "#183024" },
            { text: String(entry.blocks.length), color: "#183024" }
          ]
        }
      );
      smLabel.position.set(entry.center.x, entry.center.y + (entry.size.y * 0.5) + host.sceneData.smLabelLift, entry.center.z);
      addSpawnedObject(host, host.overlayGroup, smLabel, { startScale: 0.9, lift: 0.14, duration: 150 });

      var sharedLabel = makeSpriteLabel(
        "",
        4.8,
        0.74,
        {
          kicker: "Shared Memory",
          fill: "rgba(253, 246, 235, 0.95)",
          stroke: "rgba(204, 143, 57, 0.22)",
          kickerColor: "#b97819",
          vectorParts: [
            { text: String(entry.sharedMemoryKB) + " KB  ", color: "#183024" },
            { text: "of ", color: "#183024" },
            { text: String(getCurrentGpuProfile().sharedMemoryPerSMKB) + " KB", color: "#183024" }
          ]
        }
      );
      if (state.viewOptions.showSmMemory) {
        sharedLabel.position.set(entry.center.x + entry.size.x * 0.68, entry.center.y, entry.center.z);
        addSpawnedObject(host, host.overlayGroup, sharedLabel, { startScale: 0.9, lift: 0.14, duration: 150 });
      }
    });
  }

  function addQueueLabels(host) {
    if (!host.sceneData || !host.sceneData.queueSmEntries || host.sceneData.queueSmEntries.length === 0) {
      return;
    }

    host.sceneData.queueSmEntries.forEach(function (entry, index) {
      var queueLabel = makeSpriteLabel(
        "",
        5,
        0.82,
        {
          kicker: index === 0 ? "Queued Streaming Multiprocessor" : "Queued SM",
          fill: "rgba(255, 247, 235, 0.95)",
          stroke: "rgba(204, 143, 57, 0.22)",
          kickerColor: "#b97819",
          vectorParts: [
            { text: "QSM " + index + "  ", color: "#183024" },
            { text: "Blocks ", color: "#183024" },
            { text: String(entry.blocks.length), color: "#183024" }
          ]
        }
      );
      queueLabel.position.set(entry.center.x, entry.center.y + entry.size.y * 0.62 + host.sceneData.smLabelLift, entry.center.z);
      addSpawnedObject(host, host.overlayGroup, queueLabel, { startScale: 0.9, lift: 0.14, duration: 150 });
    });
  }

  function rebuildCombinedScene(host, launch, derived) {
    clearGroup(host.rootGroup);
    clearGroup(host.overlayGroup);

    var data = buildSceneData(launch, derived);
    host.sceneData = data;
    var dummy = new THREE.Object3D();

    var smGeometry = new THREE.BoxGeometry(data.smSizeX, data.smSizeY, data.smSizeZ);
    var smMesh = new THREE.InstancedMesh(
      smGeometry,
      new THREE.MeshStandardMaterial({
        color: 0x6f86d6,
        transparent: true,
        opacity: 0.14,
        roughness: 0.52,
        metalness: 0.02,
        depthWrite: false
      }),
      data.smEntries.length
    );
    var smMemoryMesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.3, 1, 0.3),
      new THREE.MeshStandardMaterial({
        color: 0xe0a74c,
        transparent: true,
        opacity: 0.9,
        roughness: 0.35,
        metalness: 0.04
      }),
      data.smEntries.length
    );

    data.smEntries.forEach(function (entry, index) {
      dummy.position.copy(entry.center);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      smMesh.setMatrixAt(index, dummy.matrix);

      dummy.position.set(
        entry.center.x + entry.size.x * 0.58,
        entry.center.y - entry.size.y * 0.5 + Math.max(0.2, entry.size.y * entry.sharedMemoryRatio * 0.5),
        entry.center.z
      );
      dummy.scale.set(1, Math.max(0.08, entry.size.y * Math.max(entry.sharedMemoryRatio, 0.03)), 1);
      dummy.updateMatrix();
      smMemoryMesh.setMatrixAt(index, dummy.matrix);
    });
    smMesh.instanceMatrix.needsUpdate = true;
    smMemoryMesh.instanceMatrix.needsUpdate = true;
    addSpawnedObject(host, host.rootGroup, smMesh, { startScale: 0.96, lift: 0.24, duration: 135 });
    if (state.viewOptions.showSmMemory) {
      addSpawnedObject(host, host.rootGroup, smMemoryMesh, { startScale: 0.96, lift: 0.18, duration: 135 });
    }

    if (data.queueSmEntries.length > 0) {
      var queuedSmMesh = new THREE.InstancedMesh(
        smGeometry,
        new THREE.MeshStandardMaterial({
          color: 0xf0c570,
          transparent: true,
          opacity: 0.15,
          roughness: 0.52,
          metalness: 0.02,
          depthWrite: false
        }),
        data.queueSmEntries.length
      );
      data.queueSmEntries.forEach(function (entry, index) {
        dummy.position.copy(entry.center);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        queuedSmMesh.setMatrixAt(index, dummy.matrix);
      });
      queuedSmMesh.instanceMatrix.needsUpdate = true;
      addSpawnedObject(host, host.rootGroup, queuedSmMesh, { startScale: 0.96, lift: 0.24, duration: 135 });
    }

    var shellGeometry = new THREE.BoxGeometry(data.shellSizeX, data.shellSizeY, data.shellSizeZ);
    var shellMesh = new THREE.InstancedMesh(
      shellGeometry,
      new THREE.MeshStandardMaterial({
        color: 0x5ea9b2,
        transparent: true,
        opacity: 0.22,
        roughness: 0.5,
        metalness: 0.02,
        depthWrite: false
      }),
      data.shellEntries.length
    );
    shellMesh.userData.itemType = "block";
    shellMesh.userData.instances = data.shellEntries;
    shellMesh.userData.pickPriority = 1;

    data.shellEntries.forEach(function (entry, index) {
      dummy.position.copy(entry.center);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      shellMesh.setMatrixAt(index, dummy.matrix);
    });
    shellMesh.instanceMatrix.needsUpdate = true;
    addSpawnedObject(host, host.rootGroup, shellMesh, { startScale: 0.965, lift: 0.16, duration: 125 });
    host.pickTargets = [shellMesh];

    if (data.queueShellEntries.length > 0) {
      var queuedShellMesh = new THREE.InstancedMesh(
        shellGeometry,
        new THREE.MeshStandardMaterial({
          color: 0xe0a74c,
          transparent: true,
          opacity: 0.18,
          roughness: 0.5,
          metalness: 0.02,
          depthWrite: false
        }),
        data.queueShellEntries.length
      );
      queuedShellMesh.userData.itemType = "block";
      queuedShellMesh.userData.instances = data.queueShellEntries;
      queuedShellMesh.userData.pickPriority = 1;

      data.queueShellEntries.forEach(function (entry, index) {
        dummy.position.copy(entry.center);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        queuedShellMesh.setMatrixAt(index, dummy.matrix);
      });
      queuedShellMesh.instanceMatrix.needsUpdate = true;
      addSpawnedObject(host, host.rootGroup, queuedShellMesh, { startScale: 0.965, lift: 0.16, duration: 125 });
      host.pickTargets.push(queuedShellMesh);
    }

    var threadGeometry = new THREE.BoxGeometry(data.threadUnit, data.threadUnit, data.threadUnit);
    var threadDepthMaterial = new THREE.MeshBasicMaterial();
    threadDepthMaterial.colorWrite = false;
    threadDepthMaterial.depthWrite = true;
    threadDepthMaterial.depthTest = true;
    var threadDepthMesh = new THREE.InstancedMesh(
      threadGeometry,
      threadDepthMaterial,
      data.threadEntries.length
    );
    threadDepthMesh.userData.itemType = "thread";
    threadDepthMesh.userData.instances = data.threadEntries;
    threadDepthMesh.userData.pickPriority = 0;

    var threadMesh = new THREE.InstancedMesh(
      threadGeometry,
      new THREE.MeshStandardMaterial({
        roughness: 0.42,
        metalness: 0.03,
        transparent: true,
        opacity: 0.92,
        depthWrite: false
      }),
      data.threadEntries.length
    );
    threadMesh.userData.itemType = "thread";
    threadMesh.userData.instances = data.threadEntries;
    threadMesh.userData.pickPriority = 3;

    data.threadEntries.forEach(function (entry, index) {
      dummy.position.copy(entry.position);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      threadDepthMesh.setMatrixAt(index, dummy.matrix);
      threadMesh.setMatrixAt(index, dummy.matrix);
      threadMesh.setColorAt(index, threadColor(
        entry.thread.x,
        entry.thread.y,
        entry.thread.z,
        launch.block,
        false
      ));
    });
    threadDepthMesh.instanceMatrix.needsUpdate = true;
    threadMesh.instanceMatrix.needsUpdate = true;
    if (threadMesh.instanceColor) {
      threadMesh.instanceColor.needsUpdate = true;
    }
    addSpawnedObject(host, host.rootGroup, threadDepthMesh, { startScale: 0.975, lift: 0.08, duration: 95 });
    addSpawnedObject(host, host.rootGroup, threadMesh, { startScale: 0.975, lift: 0.08, duration: 95 });

    updateHoverOverlay(host, launch);
    return data.sceneSpacing;
  }

  var combinedHost = createSceneHost(combinedViewRoot);
  threadModeButton.addEventListener("click", function () {
    state.colorMode = "thread";
    updateModeButtons();
    redraw();
  });
  warpModeButton.addEventListener("click", function () {
    state.colorMode = "warp";
    updateModeButtons();
    redraw();
  });
  showSmMemoryToggle.addEventListener("change", function (event) {
    state.viewOptions.showSmMemory = event.currentTarget.checked;
    redraw();
  });
  organizeBySmToggle.addEventListener("change", function (event) {
    state.viewOptions.organizeBySm = event.currentTarget.checked;
    redraw();
  });
  resetViewButton.addEventListener("click", function () {
    var launch = math.computeLaunch(state.config);
    var derived = getDerivedState(launch);
    setHoveredObject(null, launch, derived);
    fitCameraToScene(combinedHost, 2.15);
  });

  function updateHoverOverlay(host, launch) {
    host.hoverArtifacts = null;
    clearGroup(host.overlayGroup);
    addDimensionLabels(host, launch);
    if (state.viewOptions.organizeBySm) {
      addSmLabels(host);
      addQueueLabels(host);
    }

    var hoverArtifacts = null;
    var selectedEntry = getSelectedBlockEntry(host);
    if (selectedEntry && selectedEntry.status === "resident") {
      var selectedGeometry = new THREE.BoxGeometry(
        host.sceneData.shellSizeX + 0.1,
        host.sceneData.shellSizeY + 0.1,
        host.sceneData.shellSizeZ + 0.1
      );
      var selectedHighlight = new THREE.Mesh(
        selectedGeometry,
        new THREE.MeshBasicMaterial({
          color: 0x1f6dff,
          transparent: true,
          opacity: 0.18,
          depthWrite: false
        })
      );
      selectedHighlight.position.copy(selectedEntry.center);
      selectedHighlight.scale.setScalar(1.035);
      addSpawnedObject(host, host.overlayGroup, selectedHighlight, { startScale: 0.92, lift: 0.08, duration: 110 });
      hoverArtifacts = {
        selectedHighlight: selectedHighlight,
        selectedBaseScale: 1.035
      };
    }

    if (!state.hovered || !host.sceneData) {
      host.hoverArtifacts = hoverArtifacts;
      return;
    }

    var entry = host.sceneData.blockMap[blockKey(state.hovered.block)];
    if (!entry) {
      host.hoverArtifacts = hoverArtifacts;
      return;
    }

    var shellGeometry = new THREE.BoxGeometry(
      host.sceneData.shellSizeX + 0.06,
      host.sceneData.shellSizeY + 0.06,
      host.sceneData.shellSizeZ + 0.06
    );
    var baseShellScale = state.hovered.thread ? 1.055 : 1.035;
    var baseWireScale = state.hovered.thread ? 1.06 : 1.04;
    var baseShellOpacity = state.hovered.thread ? 0.2 : 0.14;
    var baseWireOpacity = state.hovered.thread ? 0.78 : 0.96;
    var shellHighlight = new THREE.Mesh(
      shellGeometry,
      new THREE.MeshStandardMaterial({
        color: state.hovered.status === "queued" ? 0xe0a74c : 0x2b7cff,
        transparent: true,
        opacity: baseShellOpacity,
        roughness: 0.22,
        metalness: 0.08,
        emissive: state.hovered.status === "queued" ? 0x7f5310 : 0x0e3d8a,
        emissiveIntensity: state.hovered.thread ? 0.4 : 0.32,
        depthWrite: false
      })
    );
    shellHighlight.position.copy(entry.center);
    shellHighlight.scale.setScalar(baseShellScale);
    addSpawnedObject(host, host.overlayGroup, shellHighlight, { startScale: 0.92, lift: 0.08, duration: 110 });

    var shellWire = new THREE.Mesh(
      shellGeometry.clone(),
      new THREE.MeshBasicMaterial({
        color: state.hovered.status === "queued" ? 0xe0a74c : 0x2b7cff,
        wireframe: true,
        transparent: true,
        opacity: baseWireOpacity
      })
    );
    shellWire.position.copy(entry.center);
    shellWire.scale.setScalar(baseWireScale);
    addSpawnedObject(host, host.overlayGroup, shellWire, { startScale: 0.92, lift: 0.08, duration: 110 });

    var threadHighlight = null;
    if (state.hovered.thread) {
      var threadGeometry = new THREE.BoxGeometry(
        host.sceneData.threadUnit + 0.05,
        host.sceneData.threadUnit + 0.05,
        host.sceneData.threadUnit + 0.05
      );
      var threadOffset = new THREE.Vector3(
        -((launch.block.x - 1) * host.sceneData.threadSpacing) / 2,
        -((launch.block.y - 1) * host.sceneData.threadSpacing) / 2,
        -((launch.block.z - 1) * host.sceneData.threadSpacing) / 2
      );
      threadHighlight = new THREE.Mesh(
        threadGeometry,
        new THREE.MeshStandardMaterial({
          color: 0x2b7cff,
          transparent: true,
          opacity: 0.98,
          roughness: 0.22,
          metalness: 0.12,
          emissive: 0x0f4db8,
          emissiveIntensity: 0.42
        })
      );
      threadHighlight.position.set(
        entry.center.x + threadOffset.x + state.hovered.thread.x * host.sceneData.threadSpacing,
        entry.center.y + threadOffset.y + state.hovered.thread.y * host.sceneData.threadSpacing,
        entry.center.z + threadOffset.z + state.hovered.thread.z * host.sceneData.threadSpacing
      );
      threadHighlight.scale.setScalar(1.18);
      addSpawnedObject(host, host.overlayGroup, threadHighlight, { startScale: 0.9, lift: 0.06, duration: 95 });
    }

    var label = makeSpriteLabel(
      "",
      state.hovered.thread ? 4.7 : 3.9,
      0.72,
      {
        kicker: state.hovered.thread ? "Hovered Thread" : "Hovered Block",
        fill: "rgba(245, 249, 252, 0.96)",
        stroke: state.hovered.status === "queued" ? "rgba(185, 120, 25, 0.22)" : "rgba(43, 124, 255, 0.18)",
        kickerColor: state.hovered.status === "queued" ? "#b97819" : "#2b7cff",
        vectorParts: state.hovered.thread
          ? [
              { text: "B ", color: "#183024" },
              { text: "(", color: "#183024" },
              { text: String(state.hovered.block.x), color: "#d14b45" },
              { text: ", ", color: "#183024" },
              { text: String(state.hovered.block.y), color: "#c9a227" },
              { text: ", ", color: "#183024" },
              { text: String(state.hovered.block.z), color: "#356fd6" },
              { text: ")   T ", color: "#183024" },
              { text: "(", color: "#183024" },
              { text: String(state.hovered.thread.x), color: "#d14b45" },
              { text: ", ", color: "#183024" },
              { text: String(state.hovered.thread.y), color: "#c9a227" },
              { text: ", ", color: "#183024" },
              { text: String(state.hovered.thread.z), color: "#356fd6" },
              { text: ")   Warp " + math.getWarpId(state.hovered.thread, launch.block), color: "#183024" }
            ]
          : [
              { text: "B ", color: "#183024" },
              { text: "(", color: "#183024" },
              { text: String(state.hovered.block.x), color: "#d14b45" },
              { text: ", ", color: "#183024" },
              { text: String(state.hovered.block.y), color: "#c9a227" },
              { text: ", ", color: "#183024" },
              { text: String(state.hovered.block.z), color: "#356fd6" },
              { text: ")", color: "#183024" }
            ]
      }
    );
    label.position.set(entry.center.x, entry.center.y + host.sceneData.shellSizeY * 0.72, entry.center.z);
    addSpawnedObject(host, host.overlayGroup, label, { startScale: 0.9, lift: 0.1, duration: 125 });

    host.hoverArtifacts = {
      selectedHighlight: hoverArtifacts ? hoverArtifacts.selectedHighlight : null,
      selectedBaseScale: hoverArtifacts ? hoverArtifacts.selectedBaseScale : 1.035,
      shellHighlight: shellHighlight,
      shellWire: shellWire,
      threadHighlight: threadHighlight,
      label: label,
      baseShellScale: baseShellScale,
      baseShellOpacity: baseShellOpacity,
      baseWireScale: baseWireScale,
      baseWireOpacity: baseWireOpacity,
      baseThreadScale: 1.18,
      baseLabelY: entry.center.y + host.sceneData.shellSizeY * 0.72
    };
  }

  function hostSmIndex(block) {
    if (!combinedHost.sceneData || !combinedHost.sceneData.blockMap) {
      return "-";
    }
    var entry = combinedHost.sceneData.blockMap[blockKey(block)];
    if (!entry) {
      return "-";
    }
    return entry.status === "queued" ? "Queue" : entry.sm;
  }

  function resolveThreadFromBlock(blockEntry, launch, host) {
    if (!blockEntry || blockEntry.status !== "resident" || !host || !host.sceneData) {
      return null;
    }

    var spacing = host.sceneData.threadSpacing;
    var offsetX = -((launch.block.x - 1) * spacing) / 2;
    var offsetY = -((launch.block.y - 1) * spacing) / 2;
    var offsetZ = -((launch.block.z - 1) * spacing) / 2;
    var center = blockEntry.center;
    var bestThread = null;
    var bestPixelDistanceSq = Infinity;
    var bestDepth = Infinity;
    var pointer = host.lastPointerPixels || { x: 0, y: 0, width: 1, height: 1 };
    var maxPixelDistanceSq = Math.pow(Math.max(18, host.sceneData.threadUnit * 38), 2);
    var point = new THREE.Vector3();
    var projected = new THREE.Vector3();
    var pixelX = 0;
    var pixelY = 0;

    for (var tz = 0; tz < launch.block.z; tz += 1) {
      for (var ty = 0; ty < launch.block.y; ty += 1) {
        for (var tx = 0; tx < launch.block.x; tx += 1) {
          point.set(
            center.x + offsetX + tx * spacing,
            center.y + offsetY + ty * spacing,
            center.z + offsetZ + tz * spacing
          );
          projected.copy(point).project(host.camera);
          if (projected.z < -1 || projected.z > 1) {
            continue;
          }
          pixelX = ((projected.x + 1) * 0.5) * pointer.width;
          pixelY = ((1 - projected.y) * 0.5) * pointer.height;
          var dx = pixelX - pointer.x;
          var dy = pixelY - pointer.y;
          var pixelDistanceSq = dx * dx + dy * dy;
          if (
            pixelDistanceSq <= maxPixelDistanceSq &&
            (pixelDistanceSq < bestPixelDistanceSq ||
              (Math.abs(pixelDistanceSq - bestPixelDistanceSq) < 0.0001 && projected.z < bestDepth))
          ) {
            bestPixelDistanceSq = pixelDistanceSq;
            bestDepth = projected.z;
            bestThread = { x: tx, y: ty, z: tz };
          }
        }
      }
    }

    return bestThread;
  }

  function getHoverIdleMessage(launch, residency) {
    var message =
      fmtAxisLine('Grid blocks:', launch.grid) +
      '  |  ' +
      fmtAxisLine('Block threads:', launch.block) +
      '  |  Resident = ' + residency.residentBlockCount +
      '  |  Queued = ' + residency.queuedBlockCount +
      '  |  Total launched threads = ' + launch.totalThreadsLaunched + '. ';

    if (shouldScopeThreadHover(launch)) {
      message += state.selectedBlock
        ? 'Thread hover is scoped to selected block ' + vectorHtml(state.selectedBlock) + '. Click the same block again or empty space to clear.'
        : 'Dense launch: hover blocks normally, then click one resident block to inspect only its threads.';
    } else {
      message += 'Hover a resident block or thread cube to inspect it.';
    }
    return message;
  }

  function setHoveredObject(hit, launch, derived) {
    var nextHover = null;
    if (hit && hit.object && hit.object.userData && hit.object.userData.instances && hit.instanceId != null) {
      var meta = hit.object.userData.instances[hit.instanceId];
      var canResolveThread = meta.status === "resident" && (
        !shouldScopeThreadHover(launch) || sameBlock(state.selectedBlock, meta.block)
      );
      var resolvedThread = canResolveThread && !meta.thread ? resolveThreadFromBlock(meta, launch, combinedHost) : meta.thread;
      nextHover = {
        object: hit.object,
        instanceId: hit.instanceId,
        itemType: resolvedThread ? "thread" : hit.object.userData.itemType,
        status: meta.status || "resident",
        block: {
          x: meta.block.x,
          y: meta.block.y,
          z: meta.block.z
        },
        thread: resolvedThread ? {
          x: resolvedThread.x,
          y: resolvedThread.y,
          z: resolvedThread.z
        } : null
      };
    }

    if (
      state.hovered &&
      nextHover &&
      state.hovered.object === nextHover.object &&
      state.hovered.instanceId === nextHover.instanceId &&
      state.hovered.itemType === nextHover.itemType &&
      sameBlock(state.hovered.block, nextHover.block) &&
      sameThread(state.hovered.thread, nextHover.thread)
    ) {
      return;
    }
    if (!nextHover) {
      var residency = derived.residency;
      state.hovered = null;
      renderStats(launch, derived);
      renderHierarchy(launch, derived);
      combinedInfoRoot.textContent = '';
      combinedInfoRoot.innerHTML = getHoverIdleMessage(launch, residency);
      updateHoverOverlay(combinedHost, launch);
      combinedHost.render();
      return;
    }
    state.hovered = nextHover;

    renderStats(launch, derived);
    renderHierarchy(launch, derived);

    if (state.hovered.itemType === "thread" && state.hovered.thread) {
      combinedInfoRoot.textContent =
        '';
      combinedInfoRoot.innerHTML =
        'Hovering thread in block ' + vectorHtml(state.hovered.block) +
        '  |  thread ' + vectorHtml(state.hovered.thread) +
        '  |  SM ' + hostSmIndex(state.hovered.block) +
        '  |  ' + (state.hovered.status === "queued" ? 'Queued' : 'Resident') +
        (state.colorMode === "warp"
          ? '  |  warp ' + math.getWarpId(state.hovered.thread, launch.block)
          : '') +
        '  |  Total launched threads = ' + launch.totalThreadsLaunched + '.';
    } else {
      combinedInfoRoot.textContent = '';
      combinedInfoRoot.innerHTML =
        'Hovering block ' + vectorHtml(state.hovered.block) +
        '  |  SM ' + hostSmIndex(state.hovered.block) +
        '  |  ' + (state.hovered.status === "queued" ? 'Queued' : 'Resident') +
        '  |  threads per block = ' + launch.threadsPerBlock +
        (state.colorMode === "warp"
          ? '  |  warps per block = ' + launch.warpsPerBlock
          : '') +
        (shouldScopeThreadHover(launch) && state.hovered.status === "resident"
          ? (sameBlock(state.selectedBlock, state.hovered.block)
            ? '  |  Thread hover enabled for this block'
            : '  |  Click this block to inspect its threads')
          : '') +
        '  |  Total launched threads = ' + launch.totalThreadsLaunched + '.';
    }

    updateHoverOverlay(combinedHost, launch);
    combinedHost.render();
  }

  combinedHost.renderer.domElement.addEventListener("mousemove", function (event) {
    if (combinedHost.isDragging()) {
      return;
    }
    var launch = math.computeLaunch(state.config);
    var derived = getDerivedState(launch);
    combinedHost.pick(event, function (hit) {
      setHoveredObject(hit, launch, derived);
    });
  });

  combinedHost.renderer.domElement.addEventListener("click", function (event) {
    if (combinedHost.isDragging()) {
      return;
    }
    var launch = math.computeLaunch(state.config);
    combinedHost.pick(event, function (hit) {
      if (!hit || hit.instanceId == null || !hit.object || !hit.object.userData || !hit.object.userData.instances) {
        state.selectedBlock = null;
        redraw();
        return;
      }
      var meta = hit.object.userData.instances[hit.instanceId];
      if (!meta || meta.status !== "resident") {
        state.selectedBlock = null;
        redraw();
        return;
      }
      state.selectedBlock = sameBlock(state.selectedBlock, meta.block)
        ? null
        : { x: meta.block.x, y: meta.block.y, z: meta.block.z };
      redraw();
    });
  });

  combinedHost.renderer.domElement.addEventListener("mouseleave", function () {
    var launch = math.computeLaunch(state.config);
    setHoveredObject(null, launch, getDerivedState(launch));
  });

  function renderHierarchy(launch, derived) {
    var residency = derived.residency;
    var validation = derived.validation;
    hierarchyRoot.innerHTML =
      '<article class="hierarchy-level">' +
        '<h3>Grid</h3>' +
        '<p>Explicit dispatched blocks: ' + fmt3(launch.grid) + '</p>' +
        '<p>Total blocks launched: ' + launch.totalBlocks + '</p>' +
        '<p>Execution waves: ' + residency.waveCount + '</p>' +
      '</article>' +
      '<article class="hierarchy-level">' +
        '<h3>Streaming Multiprocessor</h3>' +
        '<p>Resident SMs: ' + residency.residentSMCount + ' / ' + residency.smCount + '</p>' +
        '<p>Active blocks per SM: ' + residency.occupancy.activeBlocksPerSM + '</p>' +
        '<p>Shared memory per block: ' + state.config.kernelUsage.sharedMemoryPerBlockKB + ' KB</p>' +
      '</article>' +
      '<article class="hierarchy-level">' +
        '<h3>Block</h3>' +
        '<p>Explicit threads per block: ' + fmt3(launch.block) + '</p>' +
        '<p>Threads per block: ' + launch.threadsPerBlock + '</p>' +
        '<p>' + (state.colorMode === "warp"
          ? 'Warps per block: ' + launch.warpsPerBlock + ' (warp size ' + math.WARP_SIZE + ')'
          : 'Thread mode colors are based on thread coordinates inside the block.') + '</p>' +
      '</article>' +
      '<article class="hierarchy-level">' +
        '<h3>Scheduling</h3>' +
        '<p>Resident blocks now: ' + residency.residentBlockCount + '</p>' +
        '<p>Queued for later: ' + residency.queuedBlockCount + '</p>' +
        '<p>Launch validity: ' + (validation.isValid ? 'Valid' : validation.violations.map(function (v) { return v.code; }).join(", ")) + '</p>' +
      '</article>' +
      '<article class="hierarchy-level">' +
        '<h3>Hover</h3>' +
        '<p>Total threads launched: ' + launch.totalThreadsLaunched + '</p>' +
        '<p>Hovered block: ' + (state.hovered && state.hovered.block
          ? vectorHtml(state.hovered.block)
          : "-") + '</p>' +
      '</article>';
  }

  function redraw() {
    var sanitizedConfig = math.sanitizeConfig(state.config);
    state.config.grid = sanitizedConfig.grid;
    state.config.block = sanitizedConfig.block;
    clampKernelUsageToGpuLimits();
    if (math.computeLaunch(state.config).totalThreadsLaunched > getVisualizationThreadBudget()) {
      var fallbackConfig = lastValidConfig || {
        grid: { x: 3, y: 2, z: 2 },
        block: { x: 4, y: 4, z: 2 }
      };
      state.config.grid = fallbackConfig.grid;
      state.config.block = fallbackConfig.block;
    }
    var launch = math.computeLaunch(state.config);
    var derived = getDerivedState(launch);
    var residency = derived.residency;
    var validation = derived.validation;
    lastValidConfig = {
      grid: { x: state.config.grid.x, y: state.config.grid.y, z: state.config.grid.z },
      block: { x: state.config.block.x, y: state.config.block.y, z: state.config.block.z }
    };
    syncSelection(launch);

    renderStats(launch, derived);
    renderGpuUsagePanel(derived);
    combinedInfoRoot.textContent = '';
    combinedInfoRoot.innerHTML =
      fmtAxisLine('Grid blocks:', launch.grid) +
      '  |  ' +
      fmtAxisLine('Block threads:', launch.block) +
      '  |  Resident = ' + residency.residentBlockCount +
      '  |  Queued = ' + residency.queuedBlockCount +
      '  |  Waves = ' + residency.waveCount +
      '  |  Launch = ' + (validation.isValid ? 'Valid' : 'Invalid') +
      '  |  Layout = ' + (state.viewOptions.organizeBySm ? 'SM Organized' : 'Grid / Block') +
      '  |  Mode = ' + (state.colorMode === "warp" ? 'Warp Visualization' : 'Thread Visualization') +
      '  |  Total launched threads = ' + launch.totalThreadsLaunched +
      '. ' + (shouldScopeThreadHover(launch)
        ? (state.selectedBlock
          ? 'Selected block ' + vectorHtml(state.selectedBlock) + ' for thread hover. Drag to navigate.'
          : 'Dense launch: click one resident block to inspect its threads. Drag to navigate.')
        : 'Hover to inspect. Drag to navigate.');

    var sceneLayoutSignature = getSceneLayoutSignature();
    var shouldRefitCamera = !state._cameraInitialized || state._lastSceneLayoutSignature !== sceneLayoutSignature;
    rebuildCombinedScene(combinedHost, launch, derived);
    if (shouldRefitCamera) {
      fitCameraToScene(combinedHost, 2.15);
      state._cameraInitialized = true;
      state._lastSceneLayoutSignature = sceneLayoutSignature;
    } else {
      combinedHost.render();
    }
    renderHierarchy(launch, derived);
  }

  detectLocalVisualizationBudget();
  buildControls();
  buildGpuUsagePanel();
  updateModeButtons();
  controlsHintRoot.textContent = "Hover to inspect blocks. For dense launches, click one resident block to scope thread hover. Left drag rotates, right drag pans, mouse wheel zooms.";
  redraw();
})();





























