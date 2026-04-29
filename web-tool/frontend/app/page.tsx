'use client';

import React, { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import { 
  Play, 
  Settings, 
  Database, 
  FileText, 
  BarChart3, 
  Upload, 
  AlertCircle, 
  CheckCircle2, 
  Loader2,
  Check,
  ChevronRight,
  Download,
  Terminal,
  RefreshCw,
  Pause,
  Save,
  Cpu
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function Dashboard() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<any>(null);
  const [isUpdating, setIsUpdating] = useState(false);

  const togglePause = async () => {
    if (!jobId) return;
    try {
      await fetch(`${API_BASE}/experiment/pause/${jobId}`, { method: 'POST' });
    } catch (err) {
      console.error("Failed to toggle pause:", err);
    }
  };

  const updateAndResume = async () => {
    if (!jobId) return;
    setIsUpdating(true);
    try {
      // 1. Update config on server
      const resUpdate = await fetch(`${API_BASE}/experiment/update/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      
      if (!resUpdate.ok) throw new Error("Failed to update config");

      // 2. Resume
      await fetch(`${API_BASE}/experiment/pause/${jobId}`, { method: 'POST' });
    } catch (err) {
      console.error("Failed to update and resume:", err);
      setError("Failed to update configuration.");
    } finally {
      setIsUpdating(false);
    }
  };
  const [activeTab, setActiveTab] = useState<'config' | 'logs' | 'results' | 'analytics'>('config');
  const [error, setError] = useState<string | null>(null);
  const [scopusKey, setScopusKey] = useState<string>('');
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [showAdvancedRules, setShowAdvancedRules] = useState(false);
  const [filterPositiveOnly, setFilterPositiveOnly] = useState(false);
  const [enrichmentTask, setEnrichmentTask] = useState<any>(null);
  
  const logEndRef = useRef<HTMLDivElement>(null);

  // Default Config
  const [config, setConfig] = useState({
    nFolds: 2,
    maxGenerations: 10,
    populationSize: 30,
    crossProb: 0.9,
    mutationProb: 0.1,
    fitnessThreshold: 0.4,
    replacementStrategy: 'NEWPOPULATION',
    classificationStrategy: 'CBA',
    positiveWeight: 1.5,
    seed: 1,
    vocabStrategy: 'ALL',
    extraTerms: ''
  });

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    let interval: any;
    if (jobId && (!jobStatus || ['running', 'pending', 'paused'].includes(jobStatus.status))) {
      interval = setInterval(fetchJobStatus, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, jobStatus]);

  useEffect(() => {
    let interval: any;
    if (enrichmentTask && enrichmentTask.status === 'running') {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/enrichment/status/${enrichmentTask.task_id}`);
          const data = await res.json();
          setEnrichmentTask((prev: any) => ({ ...prev, ...data }));
          if (data.status === 'completed') {
            await fetchDatasets();
            setSelectedDataset(data.result_path);
            setTimeout(() => setEnrichmentTask(null), 3000); // Hide after 3s
          } else if (data.status === 'failed') {
            setError(data.error);
            setTimeout(() => setEnrichmentTask(null), 5000);
          }
        } catch (err) {
          console.error("Polling enrichment failed:", err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [enrichmentTask]);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [jobStatus?.logs]);

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/datasets`);
      const data = await res.json();
      setDatasets(data);
      if (data.length > 0 && !selectedDataset) {
        setSelectedDataset(data[0].path);
      }
    } catch (err) {
      setError('Could not fetch datasets. Is the backend running?');
    }
  };

  const fetchJobStatus = async () => {
    if (!jobId) return;
    try {
      const res = await fetch(`${API_BASE}/experiment/status/${jobId}`);
      const data = await res.json();
      setJobStatus(data);
      if (data.status === 'completed') {
        setActiveTab('results');
      } else if (data.status === 'failed') {
        setError(data.error);
      }
    } catch (err) {
      console.error('Error fetching job status:', err);
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPendingFile(file);
      setShowKeyModal(true);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement> | null, fileWithKey?: File, key?: string) => {
    const file = fileWithKey || e?.target.files?.[0];
    if (!file) return;
    
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    if (key || scopusKey) {
      formData.append('scopus_api_key', key || scopusKey);
    }

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      if (res.status === 401) {
        setPendingFile(file);
        setShowKeyModal(true);
        setIsUploading(false);
        return;
      }

      if (!res.ok) {
        setError(data.detail || 'Failed to upload dataset.');
        setIsUploading(false);
        return;
      }
      
      if (data.status === "enriching") {
        setEnrichmentTask({ task_id: data.task_id, status: 'running', progress: 0, current: 0, total: 0 });
        setIsUploading(false);
        setShowKeyModal(false);
        return;
      }
      
      await fetchDatasets();
      setSelectedDataset(data.path);
      setIsUploading(false);
      setShowKeyModal(false);
    } catch (err) {
      setError('Failed to upload dataset.');
      setIsUploading(false);
    }
  };

  const runExperiment = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset');
      return;
    }
    setError(null);
    setJobId(null);
    setJobStatus(null);
    setActiveTab('results');

    try {
      const res = await fetch(`${API_BASE}/experiment/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...config, datasetFilePath: selectedDataset }),
      });
      const data = await res.json();
      setJobId(data.job_id);
    } catch (err) {
      setError('Failed to start experiment.');
    }
  };

  const handleExport = () => {
    if (!jobStatus?.best_rules) return;
    const blob = new Blob([jobStatus.best_rules], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `irecs_evaluation_rules_${jobId?.slice(0, 8)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <main className="dashboard-container">
      {/* Enrichment Progress Overlay */}
      {enrichmentTask && (
        <div className="fixed bottom-8 right-8 z-50 w-96 animate-in slide-in-from-right-8 duration-300">
          <div className="bg-white/95 backdrop-blur-md border border-slate-200 rounded-2xl p-5 shadow-2xl shadow-slate-300/40">
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-accent-primary/20 flex items-center justify-center">
                  <Database className="text-accent-primary animate-pulse" size={18} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">Enriching Dataset</h3>
                  <p className="text-[10px] text-secondary/50 truncate w-40">{enrichmentTask.filename}</p>
                </div>
              </div>
              <span className="text-xs font-mono font-bold text-accent-primary">{enrichmentTask.progress}%</span>
            </div>
            
            <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden mb-3">
              <div 
                className="h-full bg-gradient-to-r from-accent-primary to-accent-primary/50 transition-all duration-500 ease-out"
                style={{ width: `${enrichmentTask.progress}%` }}
              />
            </div>
            
            <div className="flex justify-between items-center text-[10px] text-secondary/60">
              <span className="flex items-center gap-1">
                {enrichmentTask.status === 'completed' ? (
                  <><Check size={12} className="text-green-500" /> Complete</>
                ) : (
                  <><Loader2 className="animate-spin" size={12} /> Processing {enrichmentTask.current} / {enrichmentTask.total}</>
                )}
              </span>
              <span>Keep this tab open</span>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Image
              src="/logo_hu_7a1e9deccce3728a.png"
              alt="IRECS logo"
              width={40}
              height={40}
              className="logo-image"
              priority
            />
            <h1>IRECS <span>Lab</span></h1>
          </div>
        </div>

        <nav className="sidebar-nav">
          <section className="nav-group">
            <h2 className="group-title">Experiment Setup</h2>
            
            <div className="form-item">
              <label><Database size={16} /> Dataset</label>
              <select 
                value={selectedDataset} 
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="input-select"
              >
                {datasets.map((d, i) => (
                  <option key={i} value={d.path}>{d.name} ({d.source})</option>
                ))}
              </select>
              <div className="upload-wrapper">
                <input type="file" id="ds-upload" onChange={onFileChange} hidden accept=".csv" />
                <label htmlFor="ds-upload" className="btn-upload">
                  {isUploading ? <Loader2 className="animate-spin" size={14} /> : <Upload size={14} />}
                  Upload CSV
                </label>
              </div>
            </div>

            <div className="param-grid">
              <div className="form-item">
                <label>Folds</label>
                <input 
                  type="number" 
                  value={config.nFolds} 
                  onChange={(e) => setConfig({...config, nFolds: parseInt(e.target.value)})} 
                />
              </div>
              <div className="form-item">
                <label>Gens</label>
                <input 
                  type="number" 
                  value={config.maxGenerations} 
                  onChange={(e) => setConfig({...config, maxGenerations: parseInt(e.target.value)})} 
                />
              </div>
            </div>

            <div className="form-item">
              <label>Population Size</label>
              <input 
                type="range" min="10" max="500" step="10" 
                value={config.populationSize} 
                onChange={(e) => setConfig({...config, populationSize: parseInt(e.target.value)})} 
              />
              <span className="value-label">{config.populationSize} individuals</span>
            </div>

            <div className="form-item">
              <label>Fitness Threshold</label>
              <input 
                type="range" min="0" max="1" step="0.05" 
                value={config.fitnessThreshold} 
                onChange={(e) => setConfig({...config, fitnessThreshold: parseFloat(e.target.value)})} 
              />
              <span className="value-label">{config.fitnessThreshold.toFixed(2)} (Min. Quality)</span>
            </div>

            <div className="form-item">
              <label>Replacement Strategy</label>
              <select 
                value={config.replacementStrategy} 
                onChange={(e) => setConfig({...config, replacementStrategy: e.target.value})}
              >
                <option value="PUREGENERATIONAL">Generational</option>
                <option value="ELITIST">Elitist</option>
                <option value="NEWPOPULATION">New Population</option>
              </select>
            </div>
            
              <button 
                className={`btn-primary ${jobStatus?.status === 'running' ? 'btn-disabled cursor-not-allowed opacity-50' : ''}`} 
                onClick={runExperiment}
                disabled={jobStatus?.status === 'running'}
              >
                {jobStatus?.status === 'running' ? <Loader2 className="animate-spin" /> : <Play />}
                {jobStatus?.status === 'running' ? 'Experiment in Progress...' : 'Run Experiment'}
              </button>

              {jobStatus?.status === 'running' && (
                <div className="progress-container animate-fade-in mt-6">
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-semibold uppercase tracking-wider text-secondary">Evolution Progress</span>
                      <button 
                        onClick={togglePause}
                        className="p-1 hover:bg-slate-100 rounded-full transition-colors text-accent-primary"
                        title="Pause Experiment"
                      >
                        <Pause size={16} />
                      </button>
                    </div>
                    <span className="text-sm font-bold text-accent-primary ml-2">{jobStatus.progress || 0}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div 
                      className="progress-bar-fill" 
                      style={{ width: `${jobStatus.progress || 0}%` }}
                    />
                  </div>
                </div>
              )}

              {jobStatus?.status === 'paused' && (
                <div className="pause-overlay animate-fade-in mt-6 p-4 rounded-xl border border-yellow-500/30 bg-yellow-500/5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2 text-yellow-500">
                      <Pause size={20} />
                      <span className="font-bold uppercase tracking-tighter">Experiment Paused</span>
                    </div>
                    <div className="flex gap-2">
                       <button 
                        onClick={togglePause}
                        className="px-4 py-2 bg-yellow-500 text-black font-bold rounded-lg hover:bg-yellow-400 transition-all text-xs"
                      >
                        Resume
                      </button>
                      <button 
                        onClick={updateAndResume}
                        disabled={isUpdating}
                        className="px-4 py-2 bg-slate-900 text-white font-bold rounded-lg hover:bg-slate-800 transition-all text-xs flex items-center gap-2"
                      >
                        {isUpdating ? <Loader2 className="animate-spin" size={14} /> : <Save size={14} />}
                        Save & Resume
                      </button>
                    </div>
                  </div>
                  <p className="text-xs text-secondary leading-relaxed">
                    The experiment will wait until you resume. You can change parameters in the <strong>Configuration</strong> tab before clicking "Save & Resume".
                  </p>
                </div>
              )}
          </section>
        </nav>

        <div className="sidebar-footer">
          <div className={`status-badge ${jobId ? jobStatus?.status : 'idle'}`}>
             <div className="indicator" />
             {jobId ? `Job ${jobStatus?.status || '...'}` : 'Systems Ready'}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <section className="main-content">
        <header className="content-header">
          <div className="tabs">
            <button 
              className={`tab ${activeTab === 'config' ? 'active' : ''}`}
              onClick={() => setActiveTab('config')}
            >
              <Settings size={18} /> Configuration
            </button>
            <button 
              className={`tab ${activeTab === 'logs' ? 'active' : ''}`}
              onClick={() => setActiveTab('logs')}
              disabled={!jobId}
            >
              <Terminal size={18} /> Logs
            </button>
            <button 
              className={`tab ${activeTab === 'results' ? 'active' : ''}`}
              onClick={() => setActiveTab('results')}
              disabled={!jobId}
            >
              <CheckCircle2 size={18} /> Results
            </button>
            <button 
              className={`tab ${activeTab === 'analytics' ? 'active' : ''}`}
              onClick={() => setActiveTab('analytics')}
              disabled={!jobStatus?.fitness_history?.length}
            >
              <BarChart3 size={18} /> Analytics
            </button>
          </div>
        </header>

        <div className="content-viewport">
          {error && (
            <div className="alert error">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          {activeTab === 'config' && (
            <div className="view-config animate-fade-in">
              <div className="hero-section">
                <h2>Advanced Evolutionary Setup</h2>
                <p>Fine-tune G3P engine parameters for optimal classification results.</p>
              </div>
              
              <div className="config-grid">
                <div className="glass-card p-6">
                  <h3>Probabilities</h3>
                  <div className="form-row">
                    <label>Crossover Probability</label>
                    <input type="number" step="0.1" value={config.crossProb} onChange={(e) => setConfig({...config, crossProb: parseFloat(e.target.value)})} />
                  </div>
                  <div className="form-row">
                    <label>Mutation Probability</label>
                    <input type="number" step="0.1" value={config.mutationProb} onChange={(e) => setConfig({...config, mutationProb: parseFloat(e.target.value)})} />
                  </div>
                </div>

                <div className="glass-card p-6">
                  <h3>Classification</h3>
                  <div className="form-row">
                    <label>Strategy</label>
                    <select value={config.classificationStrategy} onChange={(e) => setConfig({...config, classificationStrategy: e.target.value})}>
                      <option value="CBA">CBA</option>
                      <option value="CMAR">CMAR</option>
                      <option value="CPAR">CPAR</option>
                    </select>
                  </div>
                  <div className="form-row">
                    <label>Positive Weight</label>
                    <input type="number" step="0.1" value={config.positiveWeight} onChange={(e) => setConfig({...config, positiveWeight: parseFloat(e.target.value)})} />
                  </div>
                </div>

                <div className="glass-card p-6 col-span-full">
                  <h3>Vocabulary & Text Mining</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-4">
                    <div className="form-item">
                      <label>Extraction Strategy</label>
                      <select 
                        value={config.vocabStrategy} 
                        onChange={(e) => setConfig({...config, vocabStrategy: e.target.value})}
                        className="input-select"
                      >
                        <option value="ALL">All Papers (Standard)</option>
                        <option value="POSITIVE">Positive Examples Only</option>
                        <option value="NEGATIVE">Negative Examples Only</option>
                      </select>
                      <p className="text-xs text-secondary mt-2">
                        Choose which papers are used to extract the relevant keywords for the grammar.
                      </p>
                    </div>
                    <div className="form-item">
                      <label>Manual Keywords (Optional)</label>
                      <textarea 
                        value={config.extraTerms} 
                        onChange={(e) => setConfig({...config, extraTerms: e.target.value})}
                        placeholder="e.g. machine learning, blockchain, iot"
                        className="input-textarea"
                        rows={3}
                      />
                      <p className="text-xs text-secondary mt-2">
                        Separate multiple terms with commas.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="view-logs animate-fade-in">
              <div className="terminal-window">
                <div className="terminal-header">
                  <div className="dots">
                    <span className="dot" />
                    <span className="dot" />
                    <span className="dot" />
                  </div>
                  <span className="terminal-title">Evolution Engine Logs</span>
                </div>
                <div className="terminal-body">
                  {jobStatus?.logs?.map((log: string, i: number) => (
                    <div key={i} className="log-line">
                      <span className="ln">{i + 1}</span>
                      <span className="content">{log}</span>
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'results' && (jobStatus?.results || jobStatus?.best_rules || jobId) && (
            <div className="view-results animate-fade-in">
              <div className="metrics-summary">
                <ResultCard label="Balanced Acc." value={jobStatus?.results ? jobStatus.results[0]?.toFixed(4) : "---"} color="var(--accent-primary)" />
                <ResultCard label="Accuracy" value={jobStatus?.results ? jobStatus.results[1]?.toFixed(4) : "---"} />
                <ResultCard label="Precision" value={jobStatus?.results ? jobStatus.results[2]?.toFixed(4) : "---"} />
                <ResultCard label="Recall" value={jobStatus?.results ? jobStatus.results[3]?.toFixed(4) : "---"} />
              </div>

              {((!jobStatus?.best_rules || !jobStatus.best_rules.includes('isCandidate')) && (jobStatus?.status === 'running' || jobStatus?.status === 'pending' || !jobStatus)) ? (
                <div className="flex-1 flex items-center justify-center py-10">
                  <div className="glass-card p-12 text-center animate-pulse-slow max-w-2xl w-full mx-auto">
                    <div className="flex flex-col items-center gap-8">
                      <div className="relative">
                        <div className="w-20 h-20 border-4 border-accent-primary/20 border-t-accent-primary rounded-full animate-spin"></div>
                        <Cpu className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-accent-primary" size={32} />
                      </div>
                      
                      <div className="space-y-4">
                        <h3 className="text-2xl font-bold tracking-tight">Analyzing Data & Evolving Rules</h3>
                        <div className="flex items-center justify-center gap-3 text-secondary italic">
                          <span className="inline-block w-2 h-2 bg-accent-primary rounded-full animate-ping"></span>
                          <p className="loading-text text-lg">The evolutionary algorithm is processing the first fold...</p>
                        </div>
                      </div>

                      <p className="text-xs text-secondary/60 max-w-sm mt-4">
                        This process may take a few seconds depending on the dataset size and hardware capabilities.
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                <div className="rules-section glass-card">
                  <div className="section-header">
                    <h3>Evolved Rules</h3>
                    
                    <div className="flex items-center gap-6">
                      <div className="flex items-center gap-4 bg-slate-100 px-5 py-2.5 rounded-2xl border border-slate-200 backdrop-blur-sm">
                        <span className={`text-[10px] font-bold uppercase tracking-widest transition-all duration-300 ${!showAdvancedRules ? 'text-accent-primary scale-110' : 'text-secondary/40'}`}>Simple</span>
                        
                        <button 
                          onClick={(e) => { e.preventDefault(); setShowAdvancedRules(!showAdvancedRules); }}
                          className={`relative w-14 h-7 rounded-full transition-all duration-500 shadow-inner ${showAdvancedRules ? 'bg-accent-primary/30 border-accent-primary/50' : 'bg-slate-300 border-slate-300'}`}
                          style={{ border: '2px solid', cursor: 'pointer' }}
                        >
                          <div 
                            className={`absolute top-0.5 w-5 h-5 rounded-full transition-all duration-500 shadow-[0_0_15px_rgba(255,255,255,0.4)]`} 
                            style={{ 
                              left: showAdvancedRules ? 'calc(100% - 1.5rem)' : '0.125rem',
                              backgroundColor: '#ffffff',
                              zIndex: 10
                            }}
                          />
                        </button>
                        
                        <span className={`text-[10px] font-bold uppercase tracking-widest transition-all duration-300 ${showAdvancedRules ? 'text-accent-primary scale-110' : 'text-secondary/40'}`}>Advanced</span>
                      </div>
                      
                      <button 
                        onClick={handleExport}
                        disabled={!jobStatus?.best_rules}
                        className="btn-secondary btn-sm flex items-center gap-2"
                      >
                        <Download size={14} /> Export
                      </button>
                    </div>
                  </div>

                  <div className="rules-toolbar flex items-center gap-4 px-6 py-3 bg-slate-50 border-b border-slate-200">
                    <div className="flex items-center gap-3">
                      <input 
                        type="checkbox" 
                        id="positive-filter"
                        checked={filterPositiveOnly} 
                        onChange={() => setFilterPositiveOnly(!filterPositiveOnly)}
                        className="accent-accent-primary w-4 h-4 cursor-pointer"
                      />
                      <label htmlFor="positive-filter" className="text-xs font-semibold text-secondary/80 cursor-pointer hover:text-slate-900 transition-colors select-none">
                        Filter by: <span className="text-accent-primary">Positive Candidate Rules only</span>
                      </label>
                    </div>
                  </div>
                  
                  <pre className="rules-display">
                    {jobStatus?.best_rules ? (
                      jobStatus.best_rules.split('\n')
                        .filter((line: string) => {
                          if (!filterPositiveOnly) return true;
                          if (line.includes('isCandidate')) {
                            return line.includes('isCandidate True');
                          }
                          return true;
                        })
                        .map((line: string) => {
                          if (showAdvancedRules) return line;
                          if (line.includes('|')) {
                            return line.split('|')[0].trim();
                          }
                          return line;
                        })
                        .join('\n')
                    ) : "No rules found yet."}
                  </pre>
                </div>

                <div className="papers-section glass-card mt-6">
                  <div className="section-header">
                    <h3>Relevant Papers Selected</h3>
                    <span className="papers-count">{jobStatus?.selected_papers?.length || 0} unique</span>
                  </div>

                  {!jobStatus?.selected_papers || jobStatus.selected_papers.length === 0 ? (
                    <p className="text-secondary text-sm">No relevant papers selected for this execution.</p>
                  ) : (
                    <div className="papers-table-wrapper">
                      <table className="papers-table">
                        <thead>
                          <tr>
                            <th>Title</th>
                            <th>Year</th>
                            <th>DOI</th>
                            <th>Link</th>
                          </tr>
                        </thead>
                        <tbody>
                          {jobStatus.selected_papers.map((paper: any, idx: number) => (
                            <tr key={`${paper.doi || paper.title}-${idx}`}>
                              <td>{paper.title || '-'}</td>
                              <td>{paper.year || '-'}</td>
                              <td>{paper.doi || '-'}</td>
                              <td>
                                {paper.pdfLink && paper.pdfLink !== 'undetermined' && paper.pdfLink !== 'nan' ? (
                                  <a href={paper.pdfLink} target="_blank" rel="noreferrer" className="paper-link">Open</a>
                                ) : (
                                  <span className="text-secondary">-</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
                </>
              )}
            </div>
          )}

          {activeTab === 'analytics' && jobStatus?.fitness_history?.length > 0 && (
            <div className="view-analytics animate-fade-in">
              <div className="glass-card p-8">
                <div className="section-header">
                  <h3>Fitness Progression</h3>
                  <div className="flex gap-4">
                    {jobStatus.fitness_history.map((_: any, i: number) => (
                      <span key={i} className="text-xs text-secondary">Fold {i+1}</span>
                    ))}
                  </div>
                </div>
                
                <div className="charts-container">
                  {jobStatus.fitness_history.map((history: number[], foldIdx: number) => (
                    <div key={foldIdx} className="chart-wrapper">
                      <p className="text-sm mb-4">Fold {foldIdx + 1}</p>
                      <svg width="100%" height="200" viewBox="0 0 400 200" preserveAspectRatio="none">
                        <path
                          d={`M ${history.map((val, i) => `${(i / (history.length - 1)) * 400},${200 - (val * 200)}`).join(' L ')}`}
                          fill="none"
                          stroke="var(--accent-primary)"
                          strokeWidth="2"
                        />
                        {/* Area under curve */}
                        <path
                          d={`M 0,200 L ${history.map((val, i) => `${(i / (history.length - 1)) * 400},${200 - (val * 200)}`).join(' L ')} L 400,200 Z`}
                          fill="url(#gradient)"
                          opacity="0.1"
                        />
                        <defs>
                          <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="var(--accent-primary)" />
                            <stop offset="100%" stopColor="transparent" />
                          </linearGradient>
                        </defs>
                      </svg>
                      <div className="chart-labels">
                        <span>Gen 0</span>
                        <span>Gen {history.length - 1}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'analytics' && (!jobStatus?.fitness_history || jobStatus.fitness_history.length === 0) && (
            <div className="view-analytics animate-fade-in">
              <div className="glass-card p-8 text-center">
                 <BarChart3 size={48} className="mx-auto mb-4 opacity-20" />
                 <h3>No Analytics Data</h3>
                 <p className="text-secondary">Run an experiment to see the fitness progression chart.</p>
              </div>
            </div>
          )}

          {showKeyModal && (
            <div className="modal-overlay">
              <div className="glass-card modal-content animate-fade-in">
                <h3>Scopus API Key Required</h3>
                <p className="text-secondary text-sm mb-6">
                  This dataset is missing metadata columns. Please provide your Scopus API Key to enable automatic enrichment.
                </p>
                <input 
                  type="password" 
                  placeholder="Enter API Key" 
                  value={scopusKey} 
                  onChange={(e) => setScopusKey(e.target.value)}
                  className="mb-4"
                  style={{ width: '100%', background: '#ffffff', border: '1px solid rgba(15,23,42,0.16)', color: 'var(--text-primary)', padding: '12px', borderRadius: '8px' }}
                />
                <div className="flex gap-4">
                  <button className="btn-primary flex-1" onClick={() => handleUpload(null, pendingFile!, scopusKey)}>
                    Submit & Enrich
                  </button>
                  <button className="btn-secondary" onClick={() => setShowKeyModal(false)}>Cancel</button>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <style jsx>{`
        .dashboard-container {
          display: flex;
          height: 100vh;
          width: 100vw;
          background: var(--bg-dark);
        }

        .sidebar {
          width: 320px;
          background: var(--bg-sidebar);
          border-right: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          z-index: 10;
        }

        .sidebar-header {
          padding: 32px 24px;
        }

        .logo {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .logo-image {
          width: 40px;
          height: 40px;
          object-fit: contain;
          border-radius: var(--radius-sm);
          border: 1px solid var(--border);
          background: #ffffff;
          padding: 4px;
        }

        .logo h1 {
          font-size: 1.5rem;
          font-weight: 700;
        }

        .logo h1 span {
          color: var(--accent-primary);
        }

        .sidebar-nav {
          flex: 1;
          padding: 0 24px;
          overflow-y: auto;
        }

        .group-title {
          font-size: 0.75rem;
          text-transform: uppercase;
          color: var(--text-secondary);
          letter-spacing: 0.1em;
          margin-bottom: 20px;
        }

        .form-item {
          margin-bottom: 24px;
        }

        .form-item label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.9rem;
          margin-bottom: 10px;
          color: var(--text-secondary);
        }

        .input-select, select, input[type="number"], input[type="text"], .input-textarea {
          width: 100%;
          background: var(--glass);
          border: 1px solid var(--border);
          color: var(--text-primary);
          padding: 10px 12px;
          border-radius: var(--radius-sm);
          transition: all 0.2s;
        }

        .input-textarea {
          resize: vertical;
          min-height: 80px;
          font-family: inherit;
        }

        .input-select:focus {
          border-color: var(--accent-primary);
          outline: none;
        }

        .upload-wrapper {
          margin-top: 10px;
        }

        .btn-upload {
          font-size: 0.8rem;
          color: var(--accent-primary);
          display: flex;
          align-items: center;
          gap: 6px;
          cursor: pointer;
        }

        .param-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
        }

        input[type="range"] {
          width: 100%;
          accent-color: var(--accent-primary);
        }

        .value-label {
          font-size: 0.8rem;
          color: var(--accent-primary);
        }

        .sidebar-footer {
          padding: 24px;
          border-top: 1px solid var(--border);
        }

        .status-badge {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 0.85rem;
          padding: 8px 16px;
          background: var(--glass);
          border-radius: 20px;
          border: 1px solid var(--border);
        }

        .indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--text-secondary);
        }

        .status-badge.running .indicator { background: var(--warning); box-shadow: 0 0 10px var(--warning); }
        .status-badge.completed .indicator { background: var(--success); box-shadow: 0 0 10px var(--success); }
        .status-badge.failed .indicator { background: var(--error); box-shadow: 0 0 10px var(--error); }

        .main-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .content-header {
          padding: 20px 40px;
          border-bottom: 1px solid var(--border);
        }

        .tabs {
          display: flex;
          gap: 32px;
        }

        .tab {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 0;
          color: var(--text-secondary);
          font-weight: 500;
          position: relative;
          transition: color 0.2s;
        }

        .tab:hover:not(:disabled) {
          color: var(--text-primary);
        }

        .tab.active {
          color: var(--accent-primary);
        }

        .tab.active::after {
          content: '';
          position: absolute;
          bottom: -21px;
          left: 0;
          width: 100%;
          height: 2px;
          background: var(--accent-primary);
          box-shadow: 0 -4px 12px var(--accent-primary);
        }

        .tab:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .content-viewport {
          flex: 1;
          padding: 40px;
          overflow-y: auto;
        }

        .alert {
          padding: 16px;
          border-radius: var(--radius-md);
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 32px;
        }

        .alert.error {
          background: rgba(239, 68, 68, 0.08);
          border: 1px solid rgba(239, 68, 68, 0.2);
          color: #b91c1c;
        }

        .hero-section {
          margin-bottom: 40px;
        }

        .hero-section h2 {
          font-size: 2rem;
          font-weight: 700;
          margin-bottom: 8px;
        }

        .hero-section p {
          color: var(--text-secondary);
        }

        .config-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 24px;
        }

        .p-6 { padding: 24px; }
        .p-8 { padding: 32px; }

        .form-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 20px;
        }

        .form-row label { color: var(--text-secondary); font-size: 0.95rem; }
        .form-row input, .form-row select { width: 120px; }

        .terminal-window {
          background: #f8fafc;
          border-radius: var(--radius-lg);
          border: 1px solid var(--border);
          overflow: hidden;
          font-family: var(--font-mono);
          box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12);
        }

        .terminal-header {
          background: #e2e8f0;
          padding: 12px 16px;
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .dots { display: flex; gap: 8px; }
        .dot { width: 10px; height: 10px; border-radius: 50%; background: #333; }
        .dot:nth-child(1) { background: #ff5f56; }
        .dot:nth-child(2) { background: #ffbd2e; }
        .dot:nth-child(3) { background: #27c93f; }

        .terminal-title { font-size: 0.75rem; color: #475569; letter-spacing: 0.05em; }

        .terminal-body {
          padding: 20px;
          height: 600px;
          overflow-y: auto;
          font-size: 0.85rem;
        }

        .log-line {
          display: flex;
          gap: 16px;
          margin-bottom: 4px;
        }

        .ln { color: #64748b; min-width: 30px; text-align: right; user-select: none; }
        .content { color: #0f172a; white-space: pre-wrap; }

        .metrics-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 24px;
          margin-bottom: 40px;
        }

        .rules-section {
          padding: 24px;
        }

        .papers-section {
          padding: 24px;
        }

        .papers-count {
          font-size: 0.8rem;
          color: var(--accent-primary);
          font-weight: 700;
          background: #dbeafe;
          border: 1px solid #bfdbfe;
          border-radius: 999px;
          padding: 6px 10px;
        }

        .papers-table-wrapper {
          overflow-x: auto;
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          background: #ffffff;
        }

        .papers-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.85rem;
        }

        .papers-table th,
        .papers-table td {
          text-align: left;
          padding: 10px 12px;
          border-bottom: 1px solid #e2e8f0;
          vertical-align: top;
        }

        .papers-table th {
          background: #f8fafc;
          color: var(--text-secondary);
          font-weight: 700;
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .papers-table tr:last-child td {
          border-bottom: none;
        }

        .paper-link {
          color: var(--accent-primary);
          text-decoration: none;
          font-weight: 600;
        }

        .paper-link:hover {
          text-decoration: underline;
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .rules-display {
          background: #f8fafc;
          padding: 20px;
          border-radius: var(--radius-sm);
          font-family: var(--font-mono);
          font-size: 0.85rem;
          color: #065f46;
          overflow-x: auto;
          white-space: pre-wrap;
          border: 1px solid var(--border);
        }

        .text-center { text-align: center; }
        .mx-auto { margin-left: auto; margin-right: auto; }
        .mb-4 { margin-bottom: 16px; }

        .charts-container {
          display: flex;
          flex-direction: column;
          gap: 40px;
        }

        .chart-wrapper {
          width: 100%;
          background: #f8fafc;
          padding: 20px;
          border-radius: var(--radius-md);
          border: 1px solid var(--border);
        }

        .chart-labels {
          display: flex;
          justify-content: space-between;
          margin-top: 10px;
          font-size: 0.7rem;
          color: var(--text-secondary);
        }

        .flex { display: flex; }
        .gap-4 { gap: 16px; }
        .text-xs { font-size: 0.75rem; }
        .text-sm { font-size: 0.875rem; }
        .mb-6 { margin-bottom: 24px; }
        .flex-1 { flex: 1; }

        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(15, 23, 42, 0.45);
          backdrop-filter: blur(4px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .modal-content {
          width: 90%;
          max-width: 400px;
          padding: 32px;
        }

        .mt-6 { margin-top: 24px; }

        .progress-bar-bg {
          height: 12px;
          background: rgba(148, 163, 184, 0.2);
          border-radius: 6px;
          border: 1px solid rgba(15,23,42,0.12);
          overflow: hidden;
        }

        .progress-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, #3b82f6, #60a5fa);
          box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
          transition: width 0.4s cubic-bezier(0.1, 0.7, 0.1, 1);
        }

        .text-accent-primary { color: var(--accent-primary); }
        .font-semibold { font-weight: 600; }
        .font-bold { font-weight: 700; }
        .uppercase { text-transform: uppercase; }
        .tracking-wider { letter-spacing: 0.05em; }
        .items-center { align-items: center; }
        .mb-3 { margin-bottom: 12px; }
        .ml-2 { margin-left: 8px; }
      `}</style>
    </main>
  );
}

function ResultCard({ label, value, color }: { label: string, value: string, color?: string }) {
  return (
    <div className="glass-card p-6" style={{ borderLeft: `4px solid ${color || 'transparent'}` }}>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '8px' }}>{label}</p>
      <p style={{ fontSize: '1.8rem', fontWeight: '700' }}>{value}</p>
    </div>
  );
}
