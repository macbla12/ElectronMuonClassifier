// ============================================================================
//  Plotter.C  -  plots for histograms produced by Analysis.C
//
//  Output: ONE multi-page <outDir>/AllPlots.pdf file (with bookmarks),
//  pages in order:
//      kinematics/    - pair pT, rapidity, mass, deltaPhi (with mass window)
//      classifier/    - RF response (with threshold and uncertainty-region lines)
//      efficiency/    - topocluster efficiency vs pT, eta, and (pT, eta)
//      matching/      - dEta_dPhi (with cone circle) and dR_best
//      features_emu/  - electron vs muon: overlaid                       
// ============================================================================

#include <TAxis.h>
#include <TCanvas.h>
#include <TEllipse.h>
#include <TError.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TMath.h>
#include <TPad.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace plt {

// ----------------------------------------------------------------------------
//  Configuration (consistent with Analysis.C)
// ----------------------------------------------------------------------------
const Color_t kColEl = kRed + 1;
const Color_t kColMu = kAzure + 2;
const double  kProbCut    = 0.5;
const double  kProbMargin = 0.15;
const double  kMassLo     = 2.9;
const double  kMassHi     = 3.2;
const double  kConeDR     = 0.8;
const int     kNYBins     = 5;
const char*   kLabel      = "J/#psi #rightarrow l^{+}l^{-} selection";

// ----------------------------------------------------------------------------
//  Utilities
// ----------------------------------------------------------------------------
std::string Uniq()
{
   static int n = 0;
   return Form("plt_%d", n++);
}

template <class T>
T* Get(TFile* f, const std::string& path)
{
   auto* h = dynamic_cast<T*>(f->Get(path.c_str()));
   if (!h) std::cerr << "Histogram not found: " << path << std::endl;
   return h;
}

void SetStyle()
{
   gROOT->SetBatch(kTRUE);
   gErrorIgnoreLevel = kWarning;
   TH1::AddDirectory(kFALSE);

   gStyle->SetOptStat(0);
   gStyle->SetOptTitle(0);
   gStyle->SetPalette(kBird);
   gStyle->SetNumberContours(100);
   gStyle->SetEndErrorSize(0);

   gStyle->SetTextFont(42);
   gStyle->SetLabelFont(42, "XYZ");
   gStyle->SetTitleFont(42, "XYZ");
   gStyle->SetLabelSize(0.045, "XYZ");
   gStyle->SetTitleSize(0.05, "XYZ");
   gStyle->SetTitleOffset(1.25, "X");
   gStyle->SetTitleOffset(1.4, "Y");

   gStyle->SetPadLeftMargin(0.14);
   gStyle->SetPadRightMargin(0.05);
   gStyle->SetPadBottomMargin(0.13);
   gStyle->SetPadTopMargin(0.06);
   gStyle->SetPadTickX(1);
   gStyle->SetPadTickY(1);
   gStyle->SetLegendBorderSize(0);
   gStyle->SetLegendFillStyle(0);
   gStyle->SetLegendFont(42);
}

const bool kSavePNG = false;  // true -> also save a separate PNG for each plot

std::vector<std::pair<TCanvas*, std::string>>& Pages()
{
   static std::vector<std::pair<TCanvas*, std::string>> pages;
   return pages;
}

// Does not save separate files; each canvas is queued as another page in a single PDF.
void Save(TCanvas* c, const std::string& dir, const std::string& name)
{
   const std::string section = dir.substr(dir.find_last_of('/') + 1);
   if (kSavePNG) {
      gSystem->mkdir(dir.c_str(), kTRUE);
      c->SaveAs((dir + "/" + name + ".png").c_str());
   }
   Pages().emplace_back(c, section + "/" + name);
}

// Save all pages to a single multi-page PDF (with bookmarks).
void WritePDF(const std::string& path)
{
   auto& p = Pages();
   if (p.empty()) { std::cerr << "No plots to save" << std::endl; return; }
   const size_t n = p.size();
   for (size_t i = 0; i < n; ++i) {
      const char* suffix = (n == 1) ? "" : (i == 0 ? "(" : (i == n - 1 ? ")" : ""));
      p[i].first->Print((path + suffix).c_str(), ("Title:" + p[i].second).c_str());
      delete p[i].first;
   }
   p.clear();
}

void Note(const std::string& s, double x = 0.17, double y = 0.88, double size = 0.042)
{
   TLatex t;
   t.SetNDC();
   t.SetTextFont(42);
   t.SetTextSize(size);
   t.DrawLatex(x, y, s.c_str());
}

double MinPositive(const TH1* h)
{
   double m = 1e30;
   for (int i = 0; i < h->GetNcells(); ++i) {
      const double v = h->GetBinContent(i);
      if (v > 0 && v < m) m = v;
   }
   return m < 1e30 ? m : 0.1;
}

std::string YTitle(const TH1* h, const std::string& unit)
{
   const double w = h->GetXaxis()->GetBinWidth(1);
   return Form("Entries / %.3g%s%s", w, unit.empty() ? "" : " ", unit.c_str());
}

void StyleLine(TH1* h, Color_t col, int marker = 20, double fillAlpha = 0.)
{
   h->SetLineColor(col);
   h->SetLineWidth(2);
   h->SetMarkerColor(col);
   h->SetMarkerStyle(marker);
   h->SetMarkerSize(0.8);
   if (fillAlpha > 0) h->SetFillColorAlpha(col, fillAlpha);
}

// ----------------------------------------------------------------------------
//  Single 1D plot
// ----------------------------------------------------------------------------
void Single1D(TFile* f, const std::string& path, const std::string& dir, const std::string& name,
              const std::string& xTitle, const std::string& unit, bool logY = false,
              Color_t col = kAzure + 2, const std::vector<double>& vlines = {},
              const std::string& note = "")
{
   auto* h0 = Get<TH1D>(f, path);
   if (!h0) return;
   auto* h = static_cast<TH1D*>(h0->Clone(Uniq().c_str()));
   h->SetTitle("");
   h->GetXaxis()->SetTitle(xTitle.c_str());
   h->GetYaxis()->SetTitle(YTitle(h, unit).c_str());
   StyleLine(h, col, 20, 0.25);

   auto* c = new TCanvas(Uniq().c_str(), "", 800, 800);
   double ymin, ymax;
   if (logY) {
      c->SetLogy();
      ymin = MinPositive(h) * 0.5;
      ymax = h->GetMaximum() * 30;
   } else {
      ymin = 0;
      ymax = h->GetMaximum() * 1.4;
   }
   h->SetMinimum(ymin);
   h->SetMaximum(ymax);
   h->Draw("HIST");
   h->Draw("E1 SAME");

   for (double x : vlines) {
      auto* l = new TLine(x, ymin, x, ymax);
      l->SetLineStyle(2);
      l->SetLineColor(kGray + 2);
      l->SetLineWidth(2);
      l->Draw();
   }
   Note(kLabel);
   if (!note.empty()) Note(note, 0.17, 0.82);
   Note(Form("Entries: %.0f", h0->GetEntries()), 0.17, note.empty() ? 0.82 : 0.76, 0.036);

   Save(c, dir, name);
}

// ----------------------------------------------------------------------------
//  2D plot
// ----------------------------------------------------------------------------
void Single2D(TFile* f, const std::string& path, const std::string& dir, const std::string& name,
              const std::string& xTitle, const std::string& yTitle, const std::string& zTitle,
              bool logZ = false, bool drawCone = false)
{
   auto* h0 = Get<TH2D>(f, path);
   if (!h0) return;
   auto* h = static_cast<TH2D*>(h0->Clone(Uniq().c_str()));
   h->SetTitle("");
   h->GetXaxis()->SetTitle(xTitle.c_str());
   h->GetYaxis()->SetTitle(yTitle.c_str());
   h->GetZaxis()->SetTitle(zTitle.c_str());
   h->GetZaxis()->SetTitleOffset(1.3);

   auto* c = new TCanvas(Uniq().c_str(), "", 800, 800);
   c->SetRightMargin(0.17);
   if (logZ) {
      c->SetLogz();
      h->SetMinimum(MinPositive(h) * 0.5);
   }
   h->Draw("COLZ");

   if (drawCone) {
      auto* e = new TEllipse(0, 0, kConeDR, kConeDR);
      e->SetFillStyle(0);
      e->SetLineStyle(2);
      e->SetLineWidth(2);
      e->SetLineColor(kRed);
      e->Draw();
      Note(Form("Cone #DeltaR = %.1f", kConeDR), 0.17, 0.88, 0.04);
   }
   Save(c, dir, name);
}

// ----------------------------------------------------------------------------
//  Electron vs muon
// ----------------------------------------------------------------------------
void Overlay(TFile* f, const std::string& name, const std::string& outName, const std::string& dir,
             const std::string& xTitle, const std::string& unit, bool logY, bool normalize,
             const std::string& note = "")
{
   auto* hE0 = Get<TH1D>(f, "electron/" + name);
   auto* hM0 = Get<TH1D>(f, "muon/" + name);
   if (!hE0 || !hM0) return;

   const double nE = hE0->GetEntries(), nM = hM0->GetEntries();
   auto* hE = static_cast<TH1D*>(hE0->Clone(Uniq().c_str()));
   auto* hM = static_cast<TH1D*>(hM0->Clone(Uniq().c_str()));
   for (TH1D* h : {hE, hM}) {
      h->SetTitle("");
      if (normalize && h->Integral() > 0) h->Scale(1. / h->Integral());
      h->GetXaxis()->SetTitle(xTitle.c_str());
      h->GetYaxis()->SetTitle(normalize ? "Normalised entries" : YTitle(h, unit).c_str());
   }
   StyleLine(hE, kColEl, 20, 0.20);
   StyleLine(hM, kColMu, 21, 0.20);

   const double ymaxData = std::max(hE->GetMaximum(), hM->GetMaximum());
   double ymin, ymax;
   if (logY) {
      ymin = std::min(MinPositive(hE), MinPositive(hM)) * 0.5;
      ymax = ymaxData * 40;
   } else {
      ymin = 0;
      ymax = ymaxData * 1.6;
   }
   for (TH1D* h : {hE, hM}) {
      h->SetMinimum(ymin);
      h->SetMaximum(ymax);
   }

   auto* c = new TCanvas(Uniq().c_str(), "", 800, 800);
   if (logY) c->SetLogy();
   hE->Draw("HIST");
   hM->Draw("HIST SAME");

   auto* leg = new TLegend(0.5, 0.74, 0.88, 0.90);
   leg->SetTextSize(0.04);
   leg->AddEntry(hE, Form("Electron (N = %.0f)", nE), "lf");
   leg->AddEntry(hM, Form("Muon (N = %.0f)", nM), "lf");
   leg->Draw();
   Note(kLabel);
   if (!note.empty()) Note(note, 0.17, 0.82);
   Save(c, dir, outName);
}

// ----------------------------------------------------------------------------
//  Topocluster efficiency
// ----------------------------------------------------------------------------
void Eff1D(TFile* f, const std::string& suffix, const std::string& dir,
           const std::string& xTitle, const std::string& unit)
{
   auto* eff0 = Get<TH1D>(f, "efficiency/eff_" + suffix);
   auto* num0 = Get<TH1D>(f, "efficiency/num_" + suffix);
   auto* den0 = Get<TH1D>(f, "efficiency/den_" + suffix);
   if (!eff0 || !num0 || !den0) return;

   // --- efficiency ---
   auto* eff = static_cast<TH1D*>(eff0->Clone(Uniq().c_str()));
   eff->SetTitle("");
   eff->GetXaxis()->SetTitle(xTitle.c_str());
   eff->GetYaxis()->SetTitle("Topocluster efficiency");
   StyleLine(eff, kBlack, 20);
   eff->SetMarkerSize(0.9);

   double lo = 1e9, hi = -1e9;
   for (int i = 1; i <= eff->GetNbinsX(); ++i) {
      if (den0->GetBinContent(i) <= 0) continue;
      lo = std::min(lo, eff->GetBinContent(i) - eff->GetBinError(i));
      hi = std::max(hi, eff->GetBinContent(i) + eff->GetBinError(i));
   }
   if (hi < lo) { lo = 0; hi = 1; }
   const double span = std::max(hi - lo, 0.1);
   const double ymin = std::max(0., lo - 0.5 * span);
   const double ymax = std::min(1.15, hi + 0.7 * span);
   eff->SetMinimum(ymin);
   eff->SetMaximum(ymax);

   auto* c = new TCanvas(Uniq().c_str(), "", 800, 800);
   eff->Draw("PE");
   if (ymax > 1.0) {
      auto* l = new TLine(eff->GetXaxis()->GetXmin(), 1., eff->GetXaxis()->GetXmax(), 1.);
      l->SetLineStyle(2);
      l->SetLineColor(kGray + 1);
      l->Draw();
   }
   Note(kLabel);
   const double total = den0->Integral() > 0 ? num0->Integral() / den0->Integral() : 0.;
   Note(Form("Integrated efficiency: %.3f", total), 0.17, 0.82, 0.038);
   Save(c, dir, "eff_" + suffix);
}

void Eff2D(TFile* f, const std::string& dir)
{
   auto* eff0 = Get<TH2D>(f, "efficiency/eff_pt_eta");
   auto* den0 = Get<TH2D>(f, "efficiency/den_pt_eta");
   if (!eff0 || !den0) return;

   auto* h = static_cast<TH2D*>(eff0->Clone(Uniq().c_str()));
   h->SetTitle("");
   double zmin = 1.;
   for (int ix = 1; ix <= h->GetNbinsX(); ++ix)
      for (int iy = 1; iy <= h->GetNbinsY(); ++iy) {
         if (den0->GetBinContent(ix, iy) > 0) zmin = std::min(zmin, h->GetBinContent(ix, iy));
         else h->SetBinContent(ix, iy, -1.);  // Empty bins are not drawn.
      }
   zmin = std::min(0.95, std::max(0., std::floor(zmin * 20.) / 20.));
   h->SetMinimum(zmin);
   h->SetMaximum(1.0);
   h->GetXaxis()->SetTitle("p_{T} [GeV]");
   h->GetYaxis()->SetTitle("#eta");
   h->GetZaxis()->SetTitle("Topocluster efficiency");
   h->GetZaxis()->SetTitleOffset(1.3);

   auto* c = new TCanvas(Uniq().c_str(), "", 800, 800);
   c->SetRightMargin(0.17);
   h->Draw("COLZ");
   Note(kLabel);
   Save(c, dir, "eff_pt_eta");

}

// ----------------------------------------------------------------------------
//  Variable table
// ----------------------------------------------------------------------------
struct Var {
   const char* name;
   const char* xTitle;
   const char* unit;
   bool        logY;
};

}  // namespace plt

// ============================================================================
//  Main function
// ============================================================================
void Plotter()
{
   using namespace plt;
   SetStyle();
   const char* inFile = "Plots/TestingControl/HistogramsTrueData.root";
   const char* outDir = "Plots/TestingControl";
   TFile* f = TFile::Open(inFile, "READ");
   if (!f || f->IsZombie()) {
      std::cerr << "Cannot open file: " << inFile << std::endl;
      return;
   }
   const std::string out = outDir;

   // ------------------------------------------------------------------ kinematics
   {
      const std::string d = out + "/kinematics";
      Single1D(f, "kinematics/pt",       d, "pair_pt",       "Pair p_{T} [GeV]", "GeV", false, kAzure + 2);
      Single1D(f, "kinematics/rapidity", d, "pair_rapidity", "Pair rapidity y",  "",    false, kAzure + 2);
      Single1D(f, "kinematics/dimass",   d, "pair_mass",     "m_{ll} [GeV]",     "GeV", false, kAzure + 2,
               {kMassLo, kMassHi});
      Single1D(f, "kinematics/deltaPhi", d, "pair_deltaPhi", "#Delta#phi [deg]", "deg", false, kAzure + 2);
   }

   // ------------------------------------------------------------------ classifier
   {
      const std::string d = out + "/classifier";
      const std::vector<double> cuts = {kProbCut - kProbMargin, kProbCut, kProbCut + kProbMargin};
      Single1D(f, "classifier/response",          d, "response",           "RF response P(#mu)", "", true,
               kViolet + 1, cuts, "All tracks with a topocluster");
      Single1D(f, "classifier/responseOneFound",  d, "response_oneFound",  "RF response P(#mu)", "", true,
               kViolet + 1, cuts, "Pairs with one matched track");
      Single1D(f, "classifier/responseAmbiguous", d, "response_ambiguous", "RF response P(#mu)", "", false,
               kViolet + 1, cuts, "Ambiguous pairs (first track)");
   }

   // ------------------------------------------------------------------ clusters_all
   const std::vector<Var> clusterVars = {
      {"count",   "Number of matched topoclusters",       "",  true},
      {"pt",      "Cluster p_{T} [GeV]",                  "GeV", false},
      {"f",       "F = p_{T}^{cl} / p_{T}^{trk}",         "",  false},
      {"emProb",  "Cluster EM probability",               "",  true},
      {"lambda",  "Cluster #lambda / 10^{3}",             "",  false},
      {"lambda2", "Cluster #lambda^{2} / 10^{3}",         "",  true},
      {"radius",  "Cluster radius^{2} / 10^{3}",          "",  true},
   };

   // ------------------------------------------------------------------ efficiency
   {
      const std::string d = out + "/efficiency";
      Eff1D(f, "pt",  d, "p_{T} [GeV]", "GeV");
      Eff1D(f, "eta", d, "#eta", "");
      Eff2D(f, d);
   }

   // ------------------------------------------------------------------ matching
   {
      const std::string d = out + "/matching";
      Single2D(f, "matching/dEta_dPhi", d, "dEta_dPhi", "#Delta#eta (track - cluster)",
               "#Delta#phi (track - cluster)", "Clusters", true, true);
      Single1D(f, "matching/dR_best", d, "dR_best", "#DeltaR(track, selected cluster)", "", false,
               kOrange + 7);
   }

   // ------------------------------------------------------------------ features_emu
   {
      const std::string d = out + "/features_emu";
      const std::vector<Var> leptonVars = {
         {"eta",       "#eta",                "",    false},
         {"energy",    "Energy [GeV]",        "GeV", false},
         {"pt",        "p_{T} [GeV]",         "GeV", false},
         {"phi",       "#phi [rad]",          "rad", false},
         {"pixelHits", "Pixel hits",          "",    false},
         {"trtHits",   "TRT hits",            "",    false},
         {"sctHits",   "SCT hits",            "",    false},
         {"pixeldEdX", "Pixel dE/dx",         "",    false},
         {"dimass",    "m_{ll} [GeV]",        "GeV", false},
      };

      // Normalized distributions
      for (const auto& v : leptonVars)
         Overlay(f, v.name, v.name, d, v.xTitle, v.unit, v.logY, /*normalize=*/true);

      for (const auto& v : clusterVars)
         Overlay(f, std::string("clu_") + v.name, std::string("clu_") + v.name, d, v.xTitle, v.unit, v.logY,
                 true);

      // Pair mass: not normalized, since the electron/muon event counts matter; also shown in |y| bins.
      Overlay(f, "dimass", "dimass_abs", d, "m_{ll} [GeV]", "GeV", false, false);
      for (int b = 0; b < kNYBins; ++b) {
         const std::string n    = Form("dimass_y%d", b);
         const std::string note = Form("|y| %.1f - %.1f", b * 0.5, b * 0.5 + 0.5);
         Overlay(f, n, n + "_abs", d, "m_{ll} [GeV]", "GeV", false, false, note);
      }
   }

   gSystem->mkdir(out.c_str(), kTRUE);
   WritePDF(out + "/AllPlots.pdf");
   f->Close();
}