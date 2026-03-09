(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
    return;
  }

  root.LaunchMath = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var MAX_GRID_DIMENSION = 32;
  var MAX_BLOCK_DIMENSION = { x: 1024, y: 1024, z: 64 };
  var MAX_TOTAL_THREADS = 1000000;
  var WARP_SIZE = 32;
  var GPU_PROFILES = {
    H100: {
      name: "NVIDIA H100 SXM",
      smCount: 120,
      maxThreadsPerSM: 2048,
      maxWarpsPerSM: 64,
      maxBlocksPerSM: 32,
      maxThreadsPerBlock: 1024,
      maxBlockDim: { x: 1024, y: 1024, z: 64 },
      registersPerSM: 65536,
      sharedMemoryPerSMKB: 228
    },
    A100: {
      name: "NVIDIA A100 SXM",
      smCount: 108,
      maxThreadsPerSM: 2048,
      maxWarpsPerSM: 64,
      maxBlocksPerSM: 32,
      maxThreadsPerBlock: 1024,
      maxBlockDim: { x: 1024, y: 1024, z: 64 },
      registersPerSM: 65536,
      sharedMemoryPerSMKB: 164
    },
    RTX4090: {
      name: "NVIDIA RTX 4090",
      smCount: 128,
      maxThreadsPerSM: 1536,
      maxWarpsPerSM: 48,
      maxBlocksPerSM: 24,
      maxThreadsPerBlock: 1024,
      maxBlockDim: { x: 1024, y: 1024, z: 64 },
      registersPerSM: 65536,
      sharedMemoryPerSMKB: 100
    }
  };

  function clampInt(value, min, max) {
    var parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return min;
    }
    parsed = Math.round(parsed);
    if (parsed < min) {
      return min;
    }
    if (parsed > max) {
      return max;
    }
    return parsed;
  }

  function sanitizeConfig(config) {
    return {
      grid: {
        x: clampInt(config.grid.x, 1, MAX_GRID_DIMENSION),
        y: clampInt(config.grid.y, 1, MAX_GRID_DIMENSION),
        z: clampInt(config.grid.z, 1, MAX_GRID_DIMENSION)
      },
      block: {
        x: clampInt(config.block.x, 1, MAX_BLOCK_DIMENSION.x),
        y: clampInt(config.block.y, 1, MAX_BLOCK_DIMENSION.y),
        z: clampInt(config.block.z, 1, MAX_BLOCK_DIMENSION.z)
      }
    };
  }

  function snapToPowerOfTwo(value, maxValue) {
    var safeMax = Math.max(1, Number(maxValue) || 1);
    var clamped = clampInt(value, 1, safeMax);
    var exponent = Math.round(Math.log2(clamped));
    var lower = Math.pow(2, Math.max(0, exponent - 1));
    var upper = Math.pow(2, exponent);
    if (upper > safeMax) {
      upper = safeMax;
    }
    if (lower > safeMax) {
      lower = safeMax;
    }
    return Math.abs(clamped - lower) < Math.abs(upper - clamped) ? lower : upper;
  }

  function powerOfTwoExponent(value, maxValue) {
    return Math.round(Math.log2(snapToPowerOfTwo(value, maxValue)));
  }

  function ceilDiv(a, b) {
    return Math.ceil(a / b);
  }

  function computeLaunch(config) {
    var safe = sanitizeConfig(config);
    var threadsPerBlock = safe.block.x * safe.block.y * safe.block.z;

    return {
      grid: safe.grid,
      block: safe.block,
      threadsPerBlock: threadsPerBlock,
      warpsPerBlock: Math.ceil(threadsPerBlock / WARP_SIZE),
      totalBlocks: safe.grid.x * safe.grid.y * safe.grid.z,
      totalThreadsLaunched:
        safe.grid.x * safe.grid.y * safe.grid.z * threadsPerBlock
    };
  }

  function isWithinThreadBudget(config) {
    return computeLaunch(config).totalThreadsLaunched <= MAX_TOTAL_THREADS;
  }

  function buildBlockList(launch, maxBlocks) {
    var blocks = [];
    var limit = typeof maxBlocks === "number" ? maxBlocks : 256;

    for (var bz = 0; bz < launch.grid.z; bz += 1) {
      for (var by = 0; by < launch.grid.y; by += 1) {
        for (var bx = 0; bx < launch.grid.x; bx += 1) {
          if (blocks.length >= limit) {
            return blocks;
          }

          blocks.push({
            block: { x: bx, y: by, z: bz },
            threadDims: { x: launch.block.x, y: launch.block.y, z: launch.block.z },
            threadCount: launch.threadsPerBlock,
            warpCount: launch.warpsPerBlock
          });
        }
      }
    }

    return blocks;
  }

  function getThreadLinearIndex(thread, blockDims) {
    return thread.x + (thread.y * blockDims.x) + (thread.z * blockDims.x * blockDims.y);
  }

  function getWarpId(thread, blockDims) {
    return Math.floor(getThreadLinearIndex(thread, blockDims) / WARP_SIZE);
  }

  function getWarpsPerBlock(blockDims) {
    return Math.ceil((blockDims.x * blockDims.y * blockDims.z) / WARP_SIZE);
  }

  function positiveLimit(value) {
    if (!Number.isFinite(value) || value <= 0) {
      return Infinity;
    }
    return Math.floor(value);
  }

  function computeOccupancy(kernelUsage, gpuProfile) {
    var block = sanitizeConfig({
      grid: { x: 1, y: 1, z: 1 },
      block: kernelUsage.block
    }).block;
    var threadsPerBlock = block.x * block.y * block.z;
    var warpsPerBlock = getWarpsPerBlock(block);
    var registersPerThread = clampInt(kernelUsage.registersPerThread, 0, 255);
    var sharedMemoryPerBlockKB = Math.max(0, Number(kernelUsage.sharedMemoryPerBlockKB) || 0);
    var registersPerBlock = registersPerThread * threadsPerBlock;

    var limiters = {
      maxBlocksPerSM: gpuProfile.maxBlocksPerSM,
      maxThreadsPerSM: positiveLimit(gpuProfile.maxThreadsPerSM / threadsPerBlock),
      maxWarpsPerSM: positiveLimit(gpuProfile.maxWarpsPerSM / warpsPerBlock),
      registersPerSM: registersPerBlock > 0
        ? positiveLimit(gpuProfile.registersPerSM / registersPerBlock)
        : gpuProfile.maxBlocksPerSM,
      sharedMemoryPerSM: sharedMemoryPerBlockKB > 0
        ? positiveLimit(gpuProfile.sharedMemoryPerSMKB / sharedMemoryPerBlockKB)
        : gpuProfile.maxBlocksPerSM
    };

    var activeBlocksPerSM = Math.max(
      0,
      Math.min(
        limiters.maxBlocksPerSM,
        limiters.maxThreadsPerSM,
        limiters.maxWarpsPerSM,
        limiters.registersPerSM,
        limiters.sharedMemoryPerSM
      )
    );
    var activeWarpsPerSM = Math.min(gpuProfile.maxWarpsPerSM, activeBlocksPerSM * warpsPerBlock);

    return {
      block: block,
      threadsPerBlock: threadsPerBlock,
      warpsPerBlock: warpsPerBlock,
      registersPerThread: registersPerThread,
      registersPerBlock: registersPerBlock,
      sharedMemoryPerBlockKB: sharedMemoryPerBlockKB,
      limiters: limiters,
      activeBlocksPerSM: activeBlocksPerSM,
      activeWarpsPerSM: activeWarpsPerSM,
      occupancyRatio: gpuProfile.maxWarpsPerSM > 0 ? activeWarpsPerSM / gpuProfile.maxWarpsPerSM : 0
    };
  }

  function computeKernelUsageLimits(block, gpuProfile) {
    var safeBlock = sanitizeConfig({
      grid: { x: 1, y: 1, z: 1 },
      block: block
    }).block;
    var threadsPerBlock = safeBlock.x * safeBlock.y * safeBlock.z;
    var registerCap = threadsPerBlock > 0
      ? Math.floor(gpuProfile.registersPerSM / threadsPerBlock)
      : 255;

    return {
      block: safeBlock,
      threadsPerBlock: threadsPerBlock,
      maxRegistersPerThread: Math.max(0, Math.min(255, registerCap)),
      maxSharedMemoryPerBlockKB: gpuProfile.sharedMemoryPerSMKB
    };
  }

  function validateLaunchConfig(launch, gpuProfile) {
    var violations = [];

    if (launch.block.x > gpuProfile.maxBlockDim.x) {
      violations.push({ code: "block_dim_x", message: "Block X exceeds GPU limit." });
    }
    if (launch.block.y > gpuProfile.maxBlockDim.y) {
      violations.push({ code: "block_dim_y", message: "Block Y exceeds GPU limit." });
    }
    if (launch.block.z > gpuProfile.maxBlockDim.z) {
      violations.push({ code: "block_dim_z", message: "Block Z exceeds GPU limit." });
    }
    if (launch.threadsPerBlock > gpuProfile.maxThreadsPerBlock) {
      violations.push({ code: "threads_per_block", message: "Threads per block exceed GPU limit." });
    }

    return {
      isValid: violations.length === 0,
      violations: violations
    };
  }

  function computeGpuResidency(launch, kernelUsage, gpuProfile) {
    var occupancy = computeOccupancy(
      {
        block: launch.block,
        registersPerThread: kernelUsage.registersPerThread,
        sharedMemoryPerBlockKB: kernelUsage.sharedMemoryPerBlockKB
      },
      gpuProfile
    );
    var blocksPerWave = occupancy.activeBlocksPerSM * gpuProfile.smCount;
    var residentBlockCount = Math.min(launch.totalBlocks, blocksPerWave);
    var queuedBlockCount = Math.max(0, launch.totalBlocks - residentBlockCount);
    var waveCount = blocksPerWave > 0 ? Math.ceil(launch.totalBlocks / blocksPerWave) : 0;
    var residentSMCount = occupancy.activeBlocksPerSM > 0
      ? Math.min(gpuProfile.smCount, Math.ceil(residentBlockCount / occupancy.activeBlocksPerSM))
      : 0;

    return {
      occupancy: occupancy,
      smCount: gpuProfile.smCount,
      blocksPerWave: blocksPerWave,
      residentBlockCount: residentBlockCount,
      queuedBlockCount: queuedBlockCount,
      waveCount: waveCount,
      residentSMCount: residentSMCount,
      residentWarpCount: residentBlockCount * occupancy.warpsPerBlock,
      queuedWarpCount: queuedBlockCount * occupancy.warpsPerBlock
    };
  }

  function computeSmPacking(launch, kernelUsage, gpuProfile) {
    var occupancy = computeOccupancy(
      {
        block: launch.block,
        registersPerThread: kernelUsage.registersPerThread,
        sharedMemoryPerBlockKB: kernelUsage.sharedMemoryPerBlockKB
      },
      gpuProfile
    );
    var blocks = buildBlockList(launch, launch.totalBlocks);
    var activeBlocksPerSM = Math.max(1, occupancy.activeBlocksPerSM || 0);
    var sms = [];
    var index;

    for (index = 0; index < blocks.length; index += activeBlocksPerSM) {
      var packedBlocks = blocks.slice(index, index + activeBlocksPerSM);
      sms.push({
        index: sms.length,
        blocks: packedBlocks,
        sharedMemoryKB: packedBlocks.length * occupancy.sharedMemoryPerBlockKB,
        occupancy: occupancy
      });
    }

    return {
      occupancy: occupancy,
      activeBlocksPerSM: activeBlocksPerSM,
      totalSMs: sms.length,
      sms: sms
    };
  }

  function recommendVisualizationBudget(deviceInfo) {
    var renderer = String(deviceInfo.renderer || "").toLowerCase();
    var vendor = String(deviceInfo.vendor || "").toLowerCase();
    var hardwareConcurrency = Math.max(1, Number(deviceInfo.hardwareConcurrency) || 1);

    if (!deviceInfo.webglAvailable) {
      return {
        tier: "cpu",
        maxThreads: 16000,
        label: "CPU fallback"
      };
    }

    if (
      renderer.indexOf("rtx 4090") >= 0 ||
      renderer.indexOf("rtx 4080") >= 0 ||
      renderer.indexOf("rtx 5090") >= 0 ||
      renderer.indexOf("h100") >= 0 ||
      renderer.indexOf("a100") >= 0
    ) {
      return {
        tier: "high",
        maxThreads: MAX_TOTAL_THREADS,
        label: "High-end discrete GPU"
      };
    }

    if (
      vendor.indexOf("nvidia") >= 0 ||
      renderer.indexOf("geforce") >= 0 ||
      renderer.indexOf("radeon rx") >= 0 ||
      renderer.indexOf("radeon pro") >= 0
    ) {
      return {
        tier: "discrete",
        maxThreads: 384000,
        label: "Discrete GPU"
      };
    }

    if (
      vendor.indexOf("intel") >= 0 ||
      renderer.indexOf("iris") >= 0 ||
      renderer.indexOf("uhd") >= 0 ||
      renderer.indexOf("vega") >= 0 ||
      renderer.indexOf("apple") >= 0
    ) {
      return {
        tier: "integrated",
        maxThreads: 96000,
        label: "Integrated GPU"
      };
    }

    return {
      tier: hardwareConcurrency >= 12 ? "balanced" : "cpu",
      maxThreads: hardwareConcurrency >= 12 ? 160000 : 32000,
      label: hardwareConcurrency >= 12 ? "Balanced fallback" : "Conservative fallback"
    };
  }

  function computeAdaptiveVisualizationBudget(input) {
    var currentMaxThreads = Math.max(32, Number(input.currentMaxThreads) || 32);
    var baselineMaxThreads = Math.max(32, Number(input.baselineMaxThreads) || currentMaxThreads);
    var averageFrameMs = Number(input.averageFrameMs) || 0;
    var slowFrameThresholdMs = 160;

    if (averageFrameMs < slowFrameThresholdMs || currentMaxThreads <= 32768) {
      return {
        shouldReduce: false,
        nextMaxThreads: currentMaxThreads,
        slowFrameThresholdMs: slowFrameThresholdMs
      };
    }

    var reduced = Math.max(32768, Math.floor(currentMaxThreads * 0.5));
    reduced = Math.min(reduced, baselineMaxThreads);
    reduced = reduced - (reduced % 32);

    return {
      shouldReduce: reduced < currentMaxThreads,
      nextMaxThreads: reduced,
      slowFrameThresholdMs: slowFrameThresholdMs
    };
  }

  return {
    MAX_GRID_DIMENSION: MAX_GRID_DIMENSION,
    MAX_BLOCK_DIMENSION: MAX_BLOCK_DIMENSION,
    MAX_TOTAL_THREADS: MAX_TOTAL_THREADS,
    WARP_SIZE: WARP_SIZE,
    GPU_PROFILES: GPU_PROFILES,
    sanitizeConfig: sanitizeConfig,
    snapToPowerOfTwo: snapToPowerOfTwo,
    powerOfTwoExponent: powerOfTwoExponent,
    computeLaunch: computeLaunch,
    buildBlockList: buildBlockList,
    ceilDiv: ceilDiv,
    isWithinThreadBudget: isWithinThreadBudget,
    getThreadLinearIndex: getThreadLinearIndex,
    getWarpId: getWarpId,
    getWarpsPerBlock: getWarpsPerBlock,
    computeKernelUsageLimits: computeKernelUsageLimits,
    validateLaunchConfig: validateLaunchConfig,
    computeOccupancy: computeOccupancy,
    computeGpuResidency: computeGpuResidency,
    computeSmPacking: computeSmPacking,
    recommendVisualizationBudget: recommendVisualizationBudget,
    computeAdaptiveVisualizationBudget: computeAdaptiveVisualizationBudget
  };
});
