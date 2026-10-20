// ============================================================================
//  MCAnalysis.C
//  Select lepton pairs (J/psi -> ee / mumu + backgrounds) from MC.
//  Per-sample output:
//    - ../Data/MLData<name>.root          : MLDataTree for machine learning
//    - Plots/Control/<name>_hists.root    : control histograms
// ============================================================================
#include <TROOT.h>
#include <TError.h>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TVector2.h>
#include <TLorentzVector.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>

#include <iostream>
#include <memory>
#include <vector>

namespace {

// ----------------------------------------------------------------------------
// Constants and selection criteria
// ----------------------------------------------------------------------------
constexpr double kMuonMass     = 0.1056583755;   // GeV
constexpr double kElectronMass = 0.00051099895;  // GeV
constexpr double kRad2Deg      = 180. / TMath::Pi();

constexpr double kMaxTrackEta = 2.5;
constexpr double kMinTrackPt  = 1.0;   // GeV
constexpr double kMaxPairPt   = 0.2;   // GeV
constexpr double kMaxPairMass = 3.5;   // GeV
constexpr double kMinPairMass = 2.0;   // GeV

constexpr double kMatchDeltaR = 0.8;   // track <-> topocluster matching
constexpr int    kLogEvery    = 300000;

// ----------------------------------------------------------------------------
// Sample description
// ----------------------------------------------------------------------------
struct Sample {
    TString file;
    TString name;
    bool    isMuon;
};

const std::vector<Sample> kSamples = {
    {"../Data/mc_jpsi_ee.root",         "MCElectronFull",       false},
    {"../Data/mc_jpsi_mumu.root",       "MCMuonFull",           true },
    {"../Data/mc_ee_background.root",   "MCElectronBackground", false},
    {"../Data/mc_mumu_background.root", "MCMuonBackground",     true },
};

// ----------------------------------------------------------------------------
// ML tree row
// ----------------------------------------------------------------------------
struct MLRow {
    Float_t pixelHits = 0, trtHits = 0, pixeldEdX = 0, sctHits = 0;
    Float_t fVariable = 0, emProb = 0, lambda = 0, lambda2 = 0, radius = 0;
    Float_t etaRange = 0;
    Bool_t  isMuon = kFALSE;
};

void BookMLBranches(TTree& t, MLRow& r)
{
    t.Branch("track_PixelHits", &r.pixelHits, "track_PixelHits/F");
    t.Branch("track_TRTHits",   &r.trtHits,   "track_TRTHits/F");
    t.Branch("track_PixeldEdX", &r.pixeldEdX, "track_PixeldEdX/F");
    t.Branch("track_SCTHits",   &r.sctHits,   "track_SCTHits/F");
    t.Branch("Cal_FVariable",   &r.fVariable, "Cal_FVariable/F");
    t.Branch("Cal_EMprop",      &r.emProb,    "Cal_EMprop/F");
    t.Branch("Cal_Lambda",      &r.lambda,    "Cal_Lambda/F");
    t.Branch("Cal_Lambda2",     &r.lambda2,   "Cal_Lambda2/F");
    t.Branch("Cal_Radius",      &r.radius,    "Cal_Radius/F");
    t.Branch("EtaRange",        &r.etaRange,  "EtaRange/F");
    t.Branch("IsMuon",          &r.isMuon,    "IsMuon/O");
}

// ----------------------------------------------------------------------------
// Calorimeter variables for the matched topocluster
// (if no match is found: found=false, all variables are -1)
// ----------------------------------------------------------------------------
struct ClusterVars {
    bool    found    = false;
    int     nMatched = 0;      // number of clusters in the dR < kMatchDeltaR cone
    Float_t pt       = 0;      // pT of the selected (highest-pT) cluster
    Float_t f = -1, em = -1, lambda = -1, lambda2 = -1, radius = -1;
};

// ----------------------------------------------------------------------------
// Branch readers
// ----------------------------------------------------------------------------
struct TrackReaders {
    TTreeReaderArray<unsigned int> n;
    TTreeReaderArray<float>        eta, phi, pt, pixeldEdX, pixelHits, sctHits, trtHits;
    TTreeReaderArray<int>          charge;

    explicit TrackReaders(TTreeReader& r)
        : n(r, "track_n"), eta(r, "track_eta"), phi(r, "track_phi"), pt(r, "track_pt"),
          pixeldEdX(r, "track_PixeldEdX"), pixelHits(r, "track_PixelHits"),
          sctHits(r, "track_SCTHits"), trtHits(r, "track_TRTHits"),
          charge(r, "track_charge") {}
};

// ----------------------------------------------------------------------------
// Histograms (one set per sample; each sample contains either electrons or muons)
// ----------------------------------------------------------------------------
struct Hists {
    // Pair
    TH1D *dimass, *pairPt, *rapidity, *dPhi;
    // Lepton
    TH1D *lepE, *lepEta, *lepPt, *lepPhi;
    // Track
    TH1D *pixelHits, *trtHits, *sctHits, *pixeldEdX;
    // Topoclusters
    TH1D *nTopo, *perp, *fVar, *emProb, *lambda, *lambda2, *radius;
    // Other
    TH1D *etaRange , *hDR;

    TH2D *hDEtaDPhi;

    explicit Hists(const TString& prefix)
    {
        auto book = [&](const char* n, const char* title, int nb, double lo, double hi) {
            return new TH1D(prefix + "_" + n, title, nb, lo, hi);
        };
        const double pi = TMath::Pi();

        dimass   = book("dimass",   "Lepton pair mass;M_{ll} [GeV];Events",        50, 2, 3.5);
        pairPt   = book("pairPt",   "Pair p_{T};p_{T}^{ll} [GeV];Events",          50, 0, 0.2);
        rapidity = book("rapidity", "Pair rapidity;y_{ll};Events",                 50, -3, 3);
        dPhi     = book("dPhi",     "#Delta#phi of tracks;#Delta#phi [deg];Events", 40, 160, 180);

        lepE   = book("lepE",   "Lepton energy;E [GeV];Leptons",    50, 0, 6);
        lepEta = book("lepEta", "Lepton #eta;#eta;Leptons",         50, -3, 3);
        lepPt  = book("lepPt",  "Lepton p_{T};p_{T} [GeV];Leptons", 50, 0, 1.8);
        lepPhi = book("lepPhi", "Lepton #phi;#phi;Leptons",         30, -pi, pi);
        lepPhi->SetMinimum(0);

        pixelHits = book("pixelHits", "Pixel hits;N_{pixel};Tracks", 11, -0.5, 10.5);
        trtHits   = book("trtHits",   "TRT hits;N_{TRT};Tracks",     51, -1, 50);
        sctHits   = book("sctHits",   "SCT hits;N_{SCT};Tracks",     12, 2.5, 14.5);
        pixeldEdX = book("pixeldEdX", "Pixel dE/dx;dE/dx;Tracks",   100, 0, 2);

        nTopo   = book("nTopo",   "Number of topoclusters;N;Leptons",           7, -0.5, 6.5);
        perp    = book("perp",    "Cluster p_{T};p_{T};Leptons",              100, 0, 3.5);
        fVar    = book("fVar",    "F variable;F;Leptons",                     100, 0, 4);
        emProb  = book("emProb",  "EM calorimeter probability;P_{EM};Leptons", 40, -0.1, 1.1);
        lambda  = book("lambda",  "Cluster #lambda;#lambda;Leptons",           40, 0, 4);
        lambda2 = book("lambda2", "Cluster #lambda_{2};#lambda_{2};Leptons",   40, 0, 600);
        radius  = book("radius",  "Cluster radius;R;Leptons",                  40, 0, 100);
        
        hDR = book("dR", "#DeltaR(track, cluster), all clusters passing the cut;#DeltaR;Clusters", 120, 0, 6);
        hDEtaDPhi = new TH2D(prefix + "_dEta_dPhi", "Track - cluster;#Delta#eta;#Delta#phi;Clusters",
                               60, -kMatchDeltaR, kMatchDeltaR, 60, -kMatchDeltaR, kMatchDeltaR);

        etaRange = book("etaRange", "|#eta| range;range;Leptons", 3, 0.5, 3.5);
    }

    void FillCluster(const ClusterVars& c)
    {
        nTopo->Fill(c.nMatched);
        if (!c.found) return;

        perp->Fill(c.pt);
        fVar->Fill(c.f);
        emProb->Fill(c.em);
        lambda->Fill(c.lambda / 1000);
        lambda2->Fill(c.lambda2 / 1000);
        radius->Fill(c.radius / 1000);
    }
};

struct TopoReaders {
    TTreeReaderArray<unsigned int> n;
    TTreeReaderArray<float>        eta, phi, pt, lambda, lambda2, r2, emProb;
    TTreeReaderArray<bool>         pass;

    explicit TopoReaders(TTreeReader& r)
        : n(r, "topoclus_n"), eta(r, "topo_cluster_eta"), phi(r, "topo_cluster_phi"),
          pt(r, "topo_cluster_pt"), lambda(r, "topo_cluster_lambda"),
          lambda2(r, "topo_cluster_lambda2"), r2(r, "topo_cluster_r2"),
          emProb(r, "topo_cluster_EM_prob"), pass(r, "topo_cluster_pass_sig_cut") {}

    // Finds clusters passing the signal selection within dR < kMatchDeltaR of
    // the lepton and selects the one with the highest pT.
    ClusterVars Match(const TLorentzVector& lep, Hists& h) const
    {
        ClusterVars c;
        int best = -1;

        for (int i = 0; i < int(n[0]); ++i) {
            const double dPhi = TMath::Abs(TVector2::Phi_mpi_pi(lep.Phi() - phi[i]));
            const double dEta = lep.Eta() - eta[i];
            h.hDEtaDPhi->Fill(dEta, dPhi);
            h.hDR->Fill(TMath::Sqrt(dEta * dEta + dPhi * dPhi));

            if (TMath::Sqrt(dEta * dEta + dPhi * dPhi) >= kMatchDeltaR || !pass[i]) continue;

            ++c.nMatched;
            if (pt[i] > c.pt) { c.pt = pt[i]; best = i; }
        }

        if (best >= 0) {
            c.found   = true;
            c.f       = pt[best] / lep.Perp();
            c.em      = emProb[best];
            c.lambda  = lambda[best];
            c.lambda2 = lambda2[best];
            c.radius  = r2[best];
        }
        return c;
    }
};

// 1: |eta|<1, 2: 1<=|eta|<1.5, 3: remaining range (<2.5)
inline int EtaBin(double eta)
{
    const double a = TMath::Abs(eta);
    return a < 1.0 ? 1 : (a < 1.5 ? 2 : 3);
}

// ----------------------------------------------------------------------------
// Process one sample
// ----------------------------------------------------------------------------
void ProcessSample(const Sample& s)
{
    const double mass = s.isMuon ? kMuonMass : kElectronMass;

    // --- output: ML tree ---
    std::unique_ptr<TFile> mlFile(TFile::Open(Form("../Classifiers/MLData%s.root", s.name.Data()), "RECREATE"));
    auto* mlTree = new TTree("MLDataTree", "MLDataTree");
    MLRow row;
    BookMLBranches(*mlTree, row);

    // --- output: histograms (created after cd(), so they are written to this file) ---
    std::unique_ptr<TFile> histFile(TFile::Open(Form("Plots/TrainingControl/%s_hists.root", s.name.Data()), "RECREATE"));
    histFile->cd();
    Hists h(s.name);

    // --- input ---
    TChain chain("G2TauTree");
    chain.Add(s.file);
    TTreeReader reader(&chain);
    TrackReaders trk(reader);
    TopoReaders  topo(reader);

    // For muons, reject leptons without a matched cluster.
    // This selection was previously commented out for electrons, so it remains disabled.
    const bool requireCluster = s.isMuon;

    long nEvents = 0, nLeptons = 0;

    while (reader.Next()) {
        if (++nEvents % kLogEvery == 0)
            std::cout << "Processing " << nEvents << " event..." << std::endl;

        // --- track selection ---
        if (trk.n[0] != 2) continue;

        TLorentzVector p4[2];
        bool ok = true;
        for (int i = 0; i < 2; ++i) {
            if (TMath::Abs(trk.eta[i]) < kMaxTrackEta && trk.pt[i] > kMinTrackPt)
                p4[i].SetPtEtaPhiM(trk.pt[i], trk.eta[i], trk.phi[i], mass);
            else { ok = false; break; }
        }
        if (!ok || trk.charge[0] == trk.charge[1]) continue;

        // --- pair selection ---
        const TLorentzVector pair = p4[0] + p4[1];
        if (pair.Perp() >= kMaxPairPt || pair.M() >= kMaxPairMass || pair.M() < kMinPairMass) continue;

        h.pairPt->Fill(pair.Perp());
        h.rapidity->Fill(pair.Rapidity());
        h.dimass->Fill(pair.M());
        h.dPhi->Fill(TMath::Abs(p4[0].DeltaPhi(p4[1])) * kRad2Deg);

        // --- leptons ---
        for (int i = 0; i < 2; ++i) {
            const TLorentzVector& lep = p4[i];

            h.pixeldEdX->Fill(trk.pixeldEdX[i]);
            h.pixelHits->Fill(trk.pixelHits[i]);
            h.sctHits->Fill(trk.sctHits[i]);
            h.trtHits->Fill(trk.trtHits[i]);
            h.lepE->Fill(lep.E());
            h.lepEta->Fill(lep.Eta());
            h.lepPt->Fill(lep.Pt());
            h.lepPhi->Fill(lep.Phi());

            const ClusterVars c = topo.Match(lep, h);
            h.FillCluster(c);
            if (requireCluster && !c.found) continue;

            row.pixelHits = trk.pixelHits[i];
            row.trtHits   = trk.trtHits[i];
            row.pixeldEdX = trk.pixeldEdX[i];
            row.sctHits   = trk.sctHits[i];
            row.fVariable = c.f;
            row.emProb    = c.em;
            row.lambda    = c.lambda;
            row.lambda2   = c.lambda2;
            row.radius    = c.radius;
            row.etaRange  = EtaBin(lep.Eta());
            row.isMuon    = s.isMuon;

            mlTree->Fill();
            h.etaRange->Fill(row.etaRange);
            ++nLeptons;
        }
    }

    std::cout << "===========================\n"
              << "End of file: " << s.name << "\n"
              << "Number of events: " << nEvents << "\n"
              << "Number of " << (s.isMuon ? "muons: " : "electrons: ") << nLeptons << "\n"
              << "===========================" << std::endl;

    // --- save output ---
    histFile->cd();
    histFile->Write();
    histFile->Close();

    mlFile->cd();
    mlTree->Write();
    mlFile->Close();
}

} // namespace

// ----------------------------------------------------------------------------
void TrainingMacro()
{
    gROOT->SetBatch(kTRUE);
    gErrorIgnoreLevel = kFatal;

    for (const auto& s : kSamples) ProcessSample(s);
}