use eframe::egui;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

// ── palette ────────────────────────────────────────────────────
const LIGHT: egui::Color32 = egui::Color32::from_rgb(0xf0, 0xd9, 0xb5);
const DARK:  egui::Color32 = egui::Color32::from_rgb(0xb5, 0x88, 0x63);
const HIGHLIGHT: egui::Color32 = egui::Color32::from_rgb(0xf6, 0xe5, 0x8d);
const LASTMOVE:  egui::Color32 = egui::Color32::from_rgb(0xa3, 0xd9, 0x77);
const CHECK_COL: egui::Color32 = egui::Color32::from_rgb(0xe6, 0x39, 0x46);
const BG_MAIN: egui::Color32 = egui::Color32::from_rgb(0x1a, 0x1d, 0x23);
const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(0x24, 0x28, 0x31);
const FG: egui::Color32 = egui::Color32::from_rgb(0xe6, 0xe6, 0xe6);
const MUTED: egui::Color32 = egui::Color32::from_rgb(0x9a, 0xa0, 0xa6);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x6c, 0xc6, 0xff);
const LEGAL_DOT: egui::Color32 = egui::Color32::from_rgb(0x2d, 0x6a, 0x4f);

// ── Lichess cburnett SVGs embedded ─────────────────────────────
macro_rules! svg { ($p:expr) => { include_bytes!(concat!("../assets/pieces/", $p)) } }
const W_P: &[u8] = svg!("wP.svg"); const B_P: &[u8] = svg!("bP.svg");
const W_N: &[u8] = svg!("wN.svg"); const B_N: &[u8] = svg!("bN.svg");
const W_B: &[u8] = svg!("wB.svg"); const B_B: &[u8] = svg!("bB.svg");
const W_R: &[u8] = svg!("wR.svg"); const B_R: &[u8] = svg!("bR.svg");
const W_Q: &[u8] = svg!("wQ.svg"); const B_Q: &[u8] = svg!("bQ.svg");
const W_K: &[u8] = svg!("wK.svg"); const B_K: &[u8] = svg!("bK.svg");

fn piece_svg_bytes(c: char) -> &'static [u8] {
    match c {
        'P'=>W_P,'N'=>W_N,'B'=>W_B,'R'=>W_R,'Q'=>W_Q,'K'=>W_K,
        'p'=>B_P,'n'=>B_N,'b'=>B_B,'r'=>B_R,'q'=>B_Q,'k'=>B_K, _=>W_P,
    }
}

fn raster_svg(svg: &[u8], px: u32) -> egui::ColorImage {
    let tree = {
        let mut opt = usvg::Options::default();
        opt.fontdb_mut().load_system_fonts();
        usvg::Tree::from_data(svg, &opt).ok()
    };
    if let Some(tree) = tree {
        let mut pix = tiny_skia::Pixmap::new(px, px).unwrap();
        let sx = px as f32 / 45.0; let sy = px as f32 / 45.0;
        let xf = tiny_skia::Transform::from_scale(sx, sy);
        resvg::render(&tree, xf, &mut pix.as_mut());
        let data = pix.data().to_vec();
        let rgba: Vec<u8> = data.chunks(4).flat_map(|p| [p[2],p[1],p[0],p[3]]).collect();
        egui::ColorImage::from_rgba_unmultiplied([px as usize, px as usize], &rgba)
    } else {
        egui::ColorImage::from_rgba_unmultiplied([px as usize, px as usize], &vec![0u8; (px*px*4) as usize])
    }
}

// ── UCI engine wrapper ─────────────────────────────────────────
struct UciEngine {
    child: Child,
    stdin: ChildStdin,
    console: Arc<Mutex<VecDeque<String>>>,
}
impl UciEngine {
    fn split_cmd(path: &str) -> (String, Vec<String>) {
        // allow "binary --flag arg" strings (e.g. "/usr/bin/gnuchess --uci")
        let mut parts: Vec<String> = path.split_whitespace().map(|x| x.to_string()).collect();
        if parts.is_empty() { return (path.to_string(), vec![]); }
        let bin = parts.remove(0);
        (bin, parts)
    }
    fn spawn(path: &str, console: Arc<Mutex<VecDeque<String>>>) -> Option<Self> {
        let (bin, args) = Self::split_cmd(path);
        let mut child = Command::new(&bin).args(&args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let c2 = console.clone();
        thread::spawn(move || {
            let r = BufReader::new(stdout);
            for line in r.lines().flatten() {
                let s = format!("<< {}", line);
                let mut g = c2.lock().unwrap();
                g.push_back(s); if g.len()>3000 { g.pop_front(); }
            }
        });
        let mut e = Self{ child, stdin, console };
        e.send("uci");
        Some(e)
    }
    fn send(&mut self, cmd: &str) {
        let _ = writeln!(self.stdin, "{}", cmd);
        let _ = self.stdin.flush();
        self.console.lock().unwrap().push_back(format!(">> {}", cmd));
    }
    fn kill(mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
    }
}

// ── App ────────────────────────────────────────────────────────
struct App {
    board: chess::Board,
    history: Vec<chess::Board>,
    half_moves: Vec<String>, // SAN-ish for panel
    sel: Option<chess::Square>,
    drag_from: Option<chess::Square>,
    drag_pos: Option<egui::Pos2>,
    last_move: Option<chess::ChessMove>,
    legal_targets: Vec<chess::Square>,
    owen_path: String, opp_path: String, nnue_path: String,
    mode: String, clock_choice: String,
    flip: bool, show_coords: bool, show_hints: bool,
    console: Arc<Mutex<VecDeque<String>>>,
    owen: Option<UciEngine>, opp: Option<UciEngine>,
    thinking: bool,
    eval_cp: Option<i32>, eval_mate: Option<i32>, pv: String,
    status: String,
    pending_best: Arc<Mutex<Option<String>>>,
    textures: std::collections::HashMap<char, egui::TextureHandle>,
    ready: bool,
    promo_pick: Option<(chess::Square, chess::Square)>, // from,to awaiting choice
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self{
            board: chess::Board::default(), history: vec![], half_moves: vec![],
            sel: None, drag_from: None, drag_pos: None, last_move: None, legal_targets: vec![],
            owen_path: String::new(), opp_path: String::new(), nnue_path: String::new(),
            mode:"Human vs Owen".into(), clock_choice:"5+2".into(),
            flip:false, show_coords:true, show_hints:true,
            console: Arc::new(Mutex::new(VecDeque::new())),
            owen:None, opp:None, thinking:false, eval_cp:None, eval_mate:None, pv:String::new(),
            status:"Set engine paths, then New Game.".into(),
            pending_best: Arc::new(Mutex::new(None)),
            textures: Default::default(), ready:false, promo_pick:None,
        }
    }
}

fn square_coords(sq: chess::Square, flip: bool) -> (usize, usize) {
    let f = sq.get_file().to_index(); let r = sq.get_rank().to_index();
    let df = if flip { 7-f } else { f }; let dr = if flip { r } else { 7-r };
    (df, dr)
}
fn find_king(board: &chess::Board, col: chess::Color) -> Option<chess::Square>{
    for sq in chess::ALL_SQUARES { if let Some(p)=board.piece_on(sq){ if p==chess::Piece::King && board.color_on(sq)==Some(col){ return Some(sq);} } } None
}
fn piece_char(p: chess::Piece, col: chess::Color) -> char {
    let c = match p { chess::Piece::Pawn=>'P', chess::Piece::Knight=>'N', chess::Piece::Bishop=>'B', chess::Piece::Rook=>'R', chess::Piece::Queen=>'Q', chess::Piece::King=>'K'};
    if col==chess::Color::White { c } else { c.to_ascii_lowercase() }
}
fn parse_cp(s: &str)->Option<i32>{ let i=s.find("score cp ")?; s[i+9..].split_whitespace().next()?.parse().ok() }
fn parse_mate(s: &str)->Option<i32>{ let i=s.find("score mate ")?; s[i+11..].split_whitespace().next()?.parse().ok() }
fn uci_to_move(board: &chess::Board, uci: &str) -> Option<chess::ChessMove>{
    if uci.len()<4 { return None; }
    let sf = &uci[0..2]; let st = &uci[2..4];
    let promo = if uci.len()>=5 { match uci.chars().nth(4).unwrap_or('q') { 'q'| 'Q'=>Some(chess::Piece::Queen), 'r'| 'R'=>Some(chess::Piece::Rook), 'b'| 'B'=>Some(chess::Piece::Bishop), 'n'| 'N'=>Some(chess::Piece::Knight), _=>None } } else { None };
    use std::str::FromStr;
    let f = chess::Square::from_str(sf).ok()?; let t = chess::Square::from_str(st).ok()?;
    let mv = chess::ChessMove::new(f,t,promo);
    if board.legal(mv) { Some(mv) } else { None }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut vis = egui::Visuals::dark();
        vis.window_fill = BG_MAIN; vis.panel_fill = BG_MAIN; vis.widgets.inactive.bg_fill = BG_PANEL;
        ctx.set_visuals(vis);

        if !self.ready {
            // raster lichess SVGs to textures
            for ch in ['P','N','B','R','Q','K','p','n','b','r','q','k'] {
                let bytes = piece_svg_bytes(ch);
                let img = raster_svg(bytes, 64);
                let tex = ctx.load_texture(format!("pc_{}", ch), img, egui::TextureOptions::LINEAR);
                self.textures.insert(ch, tex);
            }
            let home = std::env::var("HOME").unwrap_or_default();
            self.owen_path = format!("{}/Videos/Owen/build/owen2", home);
            self.nnue_path = format!("{}/Videos/Owen/nets/o2-v1.o2nn", home);
            let opp = format!("{}/Videos/stockfish/stockfish-ubuntu-x86-64-avx2", home);
            if std::path::Path::new(&opp).exists() { self.opp_path = opp; }
            self.ready = true;
        }

        // poll console for eval/pv/bestmove
        let mut best: Option<String> = None;
        {
            let lines: Vec<String> = self.console.lock().unwrap().iter().cloned().collect();
            for l in lines.iter().rev().take(24).rev() {
                if l.contains("score cp")  { if let Some(v)=parse_cp(l){ self.eval_cp=Some(v); self.eval_mate=None; } }
                if l.contains("score mate"){ if let Some(v)=parse_mate(l){ self.eval_mate=Some(v); self.eval_cp=None; } }
                if l.contains(" pv ") { if let Some(i)=l.find(" pv ") { self.pv = l[i+4..].trim().chars().take(96).collect(); } }
                if l.starts_with("<< bestmove") {
                    let tok = l.split_whitespace().nth(2).unwrap_or("").to_string();
                    if tok.len()>=4 && tok!="0000" { best = Some(tok); }
                    else if tok=="0000" { best=None; self.thinking=false; }
                }
            }
        }
        // consume bestmoves via pending channel too (per-engine threads push here)
        if let Some(bm) = self.pending_best.lock().unwrap().take() { best = Some(bm); }
        if let Some(uci) = best.take() {
            if let Some(mv) = uci_to_move(&self.board, &uci) {
                let san = format!("{}{}", mv.get_source(), mv.get_dest());
                self.history.push(self.board); self.half_moves.push(san);
                self.board = self.board.make_move_new(mv);
                self.last_move = Some(mv);
                self.thinking = false;
                if self.board.status()==chess::BoardStatus::Checkmate { self.status="Checkmate.".into(); }
                else if self.board.status()!=chess::BoardStatus::Ongoing { self.status=format!("Draw: {:?}", self.board.status()); }
                else {
                    let m = self.mode.clone();
                    if m=="Owen vs Opponent" {
                        let next = if self.history.len()%2==1 { "opp" } else { "owen" };
                        let _ = self.pending_best.clone();
                        self.engine_go(&next);
                    }
                }
                ctx.request_repaint();
            } else if uci=="0000" || uci=="(none)" {
                self.thinking=false;
            } else {
                self.thinking=false;
                // Try already-consumed board case: log but do not freeze
                self.status=format!("Ignored bestmove {} (illegal on {})", uci, self.board.to_string().split_whitespace().next().unwrap_or(""));
            }
        }

        // promo dialog
        if let Some((from,to)) = self.promo_pick {
            egui::Window::new("Promote").collapsible(false).resizable(false).show(ctx, |ui|{
                ui.label("Choose promotion piece");
                ui.horizontal(|ui|{
                    for (ch,pc) in [('Q',chess::Piece::Queen),('R',chess::Piece::Rook),('B',chess::Piece::Bishop),('N',chess::Piece::Knight)] {
                        if ui.button(ch.to_string()).clicked(){
                            let mv = chess::ChessMove::new(from,to,Some(pc));
                            if self.board.legal(mv){
                                self.history.push(self.board); self.half_moves.push(format!("{}{}{}", from, to, ch.to_ascii_lowercase()));
                                self.board=self.board.make_move_new(mv); self.last_move=Some(mv);
                                self.promo_pick=None; self.sel=None; self.legal_targets.clear();
                                let m=self.mode.clone();
                                if m!="Human vs Human" && m!="Analysis" {
                                    let want=if m=="Human vs Owen"{"owen"}else{"opp"};
                                    self.engine_go(want);
                                } else if m=="Analysis" { self.analyse(); }
                            }
                        }
                    }
                    if ui.button("Cancel").clicked(){ self.promo_pick=None; }
                });
            });
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui|{
            ui.horizontal(|ui|{
                ui.heading(egui::RichText::new("♚  Owen 2").size(18.0).color(FG));
                ui.label(egui::RichText::new("— Marrow Tree NNUE  •  polished board, drag or click, black console live").size(11.0).color(MUTED));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui|{
                    if ui.button("Export PGN").clicked(){ self.export_pgn(ctx); }
                    if ui.button("FEN → clipboard").clicked(){
                        let fen=self.board.to_string(); ctx.copy_text(fen.clone());
                        self.status=format!("FEN copied: {}…", &fen[..56.min(fen.len())]);
                    }
                });
            });
        });
        egui::TopBottomPanel::top("engbar").show(ctx, |ui|{
            ui.horizontal(|ui|{
                ui.label("Owen:"); let mut p=self.owen_path.clone();
                if ui.text_edit_singleline(&mut p).changed(){ self.owen_path=p.clone(); }
                if ui.button("…").clicked(){ if let Some(f)=rfd::FileDialog::new().pick_file(){ self.owen_path=f.display().to_string(); } }
                ui.label("Opp:"); let mut q=self.opp_path.clone();
                if ui.text_edit_singleline(&mut q).changed(){ self.opp_path=q.clone(); }
                if ui.button("…").clicked(){ if let Some(f)=rfd::FileDialog::new().pick_file(){ self.opp_path=f.display().to_string(); } }
                ui.label("NNUE:"); let mut n=self.nnue_path.clone();
                if ui.text_edit_singleline(&mut n).changed(){ self.nnue_path=n.clone(); }
                if ui.button("…").clicked(){ if let Some(f)=rfd::FileDialog::new().pick_file(){ self.nnue_path=f.display().to_string(); } }
            });
        });
        egui::TopBottomPanel::top("ctrl").show(ctx, |ui|{
            ui.horizontal(|ui|{
                egui::ComboBox::from_label("Mode").selected_text(&self.mode).show_ui(ui, |ui|{
                    for m in ["Human vs Owen","Human vs Opponent","Owen vs Opponent","Human vs Human","Analysis"] {
                        ui.selectable_value(&mut self.mode, m.to_string(), m);
                    }
                });
                egui::ComboBox::from_label("Clock").selected_text(&self.clock_choice).show_ui(ui, |ui|{
                    for c in ["no clock","1+0","3+2","5+2","10+5","15+10 rapid","30+0","40/90 classical"] {
                        ui.selectable_value(&mut self.clock_choice, c.to_string(), c);
                    }
                });
                ui.checkbox(&mut self.flip, "Flip");
                ui.checkbox(&mut self.show_coords, "Coords");
                ui.checkbox(&mut self.show_hints, "Hints");
                if ui.add(egui::Button::new(egui::RichText::new("▶ New Game").color(egui::Color32::BLACK)).fill(ACCENT)).clicked(){
                    self.new_game();
                }
                if ui.button("↩ Undo").clicked(){ self.undo(); }
                if ui.button("■ Stop").clicked(){ self.stop_engines(); }
            });
        });
        egui::TopBottomPanel::bottom("console").min_height(150.0).show(ctx, |ui|{
            ui.horizontal(|ui|{
                ui.label(egui::RichText::new("● Black console — live UCI   >> sent   << engine   (world-class Rust)").size(10.0).color(egui::Color32::from_rgb(0x9a,0xff,0x9a)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui|{
                    if ui.button("Clear").clicked(){ self.console.lock().unwrap().clear(); }
                    if ui.button("Copy").clicked(){
                        let s=self.console.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n");
                        ctx.copy_text(s);
                    }
                });
            });
            let bg = egui::Color32::from_rgb(0x00,0x00,0x00);
            egui::Frame::default().fill(bg).rounding(6.0).inner_margin(egui::Margin::symmetric(6.0, 6.0)).show(ui, |ui|{
                egui::ScrollArea::vertical().auto_shrink([false,false]).stick_to_bottom(true).show(ui, |ui|{
                    let lines=self.console.lock().unwrap().clone();
                    ui.style_mut().visuals.override_text_color = Some(egui::Color32::from_rgb(0xd0,0xd0,0xd0));
                    for l in lines.iter().rev().take(600).rev() {
                        let col = if l.starts_with(">>") { egui::Color32::from_rgb(0x7a,0xfc,0xff) }
                            else if l.starts_with("!!") { egui::Color32::from_rgb(0xff,0x6b,0x6b) }
                            else if l.starts_with("==") { egui::Color32::from_rgb(0xff,0xd8,0x6b) }
                            else { egui::Color32::from_rgb(0xd0,0xd0,0xd0) };
                        ui.label(egui::RichText::new(l).size(11.0).color(col).monospace());
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui|{
            ui.horizontal(|ui|{
                // eval bar 18 x 520
                let bar_w=18.0; let h=520.0;
                let (rect, _) = ui.allocate_exact_size(egui::vec2(bar_w, h), egui::Sense::hover());
                let pr=ui.painter();
                pr.rect_filled(rect, 4.0, egui::Color32::from_rgb(0x2b,0x2b,0x2b));
                if let Some(m)=self.eval_mate {
                    if m>0 { pr.rect_filled(egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, rect.center().y)), 4.0, egui::Color32::WHITE); }
                    else { pr.rect_filled(egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.center().y), rect.max), 4.0, egui::Color32::from_rgb(0x18,0x18,0x18)); }
                } else if let Some(cp)=self.eval_cp {
                    let c=cp.clamp(-600,600) as f32; let mid=rect.center().y;
                    let span=h/2.0-8.0; let dh=c/600.0*span;
                    if dh>=0.0 {
                        pr.rect_filled(egui::Rect::from_min_max(egui::pos2(rect.min.x, mid-dh), egui::pos2(rect.max.x, mid)), 4.0, egui::Color32::WHITE);
                    } else {
                        pr.rect_filled(egui::Rect::from_min_max(egui::pos2(rect.min.x, mid), egui::pos2(rect.max.x, mid-dh)), 4.0, egui::Color32::from_rgb(0x18,0x18,0x18));
                    }
                    pr.line_segment([egui::pos2(rect.min.x, mid), egui::pos2(rect.max.x, mid)], egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3a,0x3f,0x4b)));
                }
                ui.add_space(6.0);
                let size=520.0; let sq=size/8.0;
                let (board_rect, resp) = ui.allocate_exact_size(egui::vec2(size,size), egui::Sense::click_and_drag());
                let p=ui.painter_at(board_rect);
                // beveled frame
                p.rect_filled(board_rect.expand(6.0), 10.0, egui::Color32::from_rgb(0x2b,0x30,0x3a));
                p.rect_stroke(board_rect.expand(6.0), 10.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3a,0x42,0x50)));
                // squares
                for r in 0..8 { for f in 0..8 {
                    let x0=board_rect.min.x + f as f32*sq; let y0=board_rect.min.y + r as f32*sq;
                    let col=if (r+f)%2==0 { LIGHT } else { DARK };
                    let rad = if (r==0&&f==0) { 6.0 } else if (r==0&&f==7) { 6.0 } else if (r==7&&f==0) { 6.0 } else if (r==7&&f==7){6.0} else {0.0};
                    // simple: no per-corner radius, fill all then overlay border handles corners; keep square
                    p.rect_filled(egui::Rect::from_min_max(egui::pos2(x0,y0), egui::pos2(x0+sq,y0+sq)), 0.0, col);
                    let _=rad;
                }}
                // last move
                if let Some(m)=self.last_move {
                    for sq2 in [m.get_source(), m.get_dest()] {
                        let (df,dr)=square_coords(sq2, self.flip);
                        let x0=board_rect.min.x + df as f32*sq; let y0=board_rect.min.y + dr as f32*sq;
                        p.rect_filled(egui::Rect::from_min_max(egui::pos2(x0,y0), egui::pos2(x0+sq,y0+sq)), 0.0, LASTMOVE.linear_multiply(0.55));
                    }
                }
                // check ring
                if self.board.checkers().popcnt()>0 {
                    if let Some(k)=find_king(&self.board, self.board.side_to_move()){
                        let (df,dr)=square_coords(k, self.flip);
                        let x0=board_rect.min.x + df as f32*sq; let y0=board_rect.min.y + dr as f32*sq;
                        p.rect_stroke(egui::Rect::from_min_max(egui::pos2(x0,y0), egui::pos2(x0+sq,y0+sq)), 0.0, egui::Stroke::new(3.0, CHECK_COL));
                    }
                }
                // selection + hints
                if let Some(s)=self.sel {
                    let (df,dr)=square_coords(s, self.flip);
                    let x0=board_rect.min.x + df as f32*sq; let y0=board_rect.min.y + dr as f32*sq;
                    p.rect_stroke(egui::Rect::from_min_max(egui::pos2(x0+1.0,y0+1.0), egui::pos2(x0+sq-1.0,y0+sq-1.0)), 0.0, egui::Stroke::new(3.5, HIGHLIGHT));
                    if self.show_hints {
                        for t in &self.legal_targets.clone() {
                            let (df2,dr2)=square_coords(*t, self.flip);
                            let x0=board_rect.min.x + df2 as f32*sq; let y0=board_rect.min.y + dr2 as f32*sq;
                            let is_cap=self.board.piece_on(*t).is_some();
                            if is_cap { p.circle_stroke(egui::pos2(x0+sq/2.0,y0+sq/2.0), sq/2.0-5.0, egui::Stroke::new(3.0, LEGAL_DOT)); }
                            else { p.circle_filled(egui::pos2(x0+sq/2.0,y0+sq/2.0), 7.0, LEGAL_DOT); }
                        }
                    }
                }
                // pieces (skip dragged)
                for sq2 in chess::ALL_SQUARES {
                    if Some(sq2)==self.drag_from { continue; }
                    if let Some(pc)=self.board.piece_on(sq2){
                        let col=self.board.color_on(sq2).unwrap();
                        let ch=piece_char(pc,col);
                        if let Some(tex)=self.textures.get(&ch){
                            let (df,dr)=square_coords(sq2, self.flip);
                            let x0=board_rect.min.x + df as f32*sq; let y0=board_rect.min.y + dr as f32*sq;
                            let r=egui::Rect::from_min_max(egui::pos2(x0+1.0,y0+1.0), egui::pos2(x0+sq-1.0,y0+sq-1.0));
                            // subtle drop shadow
                            p.rect_filled(egui::Rect::from_min_max(egui::pos2(r.min.x+1.0, r.min.y+2.0), egui::pos2(r.max.x+1.0, r.max.y+2.0)), 0.0, egui::Color32::from_black_alpha(40));
                            p.image(tex.id(), r, egui::Rect::from_min_max(egui::pos2(0.0,0.0), egui::pos2(1.0,1.0)), egui::Color32::WHITE);
                        }
                    }
                }
                // dragged
                if let (Some(sq2), Some(pos)) = (self.drag_from, self.drag_pos) {
                    if let Some(pc)=self.board.piece_on(sq2){
                        let col=self.board.color_on(sq2).unwrap();
                        let ch=piece_char(pc,col);
                        if let Some(tex)=self.textures.get(&ch){
                            let r=egui::Rect::from_center_size(pos, egui::vec2(sq-6.0,sq-6.0));
                            p.image(tex.id(), r, egui::Rect::from_min_max(egui::pos2(0.0,0.0), egui::pos2(1.0,1.0)), egui::Color32::WHITE);
                        }
                    }
                }
                // coords
                if self.show_coords {
                    for i in 0..8 {
                        let fch = if self.flip { (b'a'+(7-i) as u8) as char } else { (b'a'+i as u8) as char };
                        let rch = if self.flip { (b'1'+i as u8) as char } else { (b'8'-i as u8) as char };
                        p.text(egui::pos2(board_rect.min.x + i as f32*sq + sq-7.0, board_rect.max.y-4.0), egui::Align2::RIGHT_BOTTOM, fch.to_string(), egui::FontId::monospace(9.0), egui::Color32::from_rgb(0x30,0x30,0x30));
                        p.text(egui::pos2(board_rect.min.x+5.0, board_rect.min.y + i as f32*sq + 7.0), egui::Align2::LEFT_TOP, rch.to_string(), egui::FontId::monospace(9.0), egui::Color32::from_rgb(0x30,0x30,0x30));
                    }
                }
                // input
                if (resp.clicked() || resp.drag_started()) && self.promo_pick.is_none() {
                    if let Some(pos)=resp.interact_pointer_pos(){
                        let f=((pos.x-board_rect.min.x)/sq).floor() as i32;
                        let r=((pos.y-board_rect.min.y)/sq).floor() as i32;
                        if (0..8).contains(&f) && (0..8).contains(&r){
                            let df= if self.flip {7-f}else{f}; let dr= if self.flip {r}else{7-r};
                            let sq2=chess::Square::make_square(chess::Rank::from_index(dr as usize), chess::File::from_index(df as usize));
                            self.handle_click(sq2, pos, resp.drag_started());
                        }
                    }
                }
                if resp.dragged(){ if let Some(pos)=resp.interact_pointer_pos(){ self.drag_pos=Some(pos); } }
                if resp.drag_stopped(){
                    if let Some(pos)=resp.interact_pointer_pos(){
                        let f=((pos.x-board_rect.min.x)/sq).floor() as i32;
                        let r=((pos.y-board_rect.min.y)/sq).floor() as i32;
                        if (0..8).contains(&f) && (0..8).contains(&r){
                            let df= if self.flip {7-f}else{f}; let dr= if self.flip {r}else{7-r};
                            let sq2=chess::Square::make_square(chess::Rank::from_index(dr as usize), chess::File::from_index(df as usize));
                            self.handle_drop(sq2);
                        } else { self.drag_from=None; self.drag_pos=None; }
                    }
                }
                // side
                ui.vertical(|ui|{
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(&self.status).size(11.0).color(MUTED));
                    ui.add_space(4.0);
                    let ev = if let Some(m)=self.eval_mate { format!("mate {:+}", m) } else if let Some(cp)=self.eval_cp { format!("{:+.2}", cp as f32/100.0) } else { "—".into() };
                    ui.label(egui::RichText::new(format!("eval {}", ev)).size(13.0).color(FG).strong());
                    ui.label(egui::RichText::new(self.pv.clone()).size(10.0).color(MUTED));
                    ui.separator();
                    ui.label(egui::RichText::new("Moves").size(11.0).color(MUTED));
                    egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui|{
                        // SAN-like using half_moves
                        let mut row = String::new();
                        for (i, m) in self.half_moves.iter().enumerate(){
                            if i%2==0 { row.push_str(&format!("{}. {} ", i/2+1, m)); } else { row.push_str(&format!("{} ", m)); }
                            if row.len()>48 { ui.label(egui::RichText::new(row.clone()).size(11.0).color(FG).monospace()); row.clear(); }
                        }
                        if !row.is_empty(){ ui.label(egui::RichText::new(row).size(11.0).color(FG).monospace()); }
                        if self.half_moves.is_empty(){ ui.label(egui::RichText::new("(no moves yet)").size(11.0).color(MUTED)); }
                        if ui.button("Copy FEN").clicked(){ let fen=self.board.to_string(); ctx.copy_text(fen.clone()); self.status=format!("FEN copied"); self.console.lock().unwrap().push_back(format!("FEN {}", fen)); }
                    });
                    ui.separator();
                    if ui.button("Analyse position (Owen)").clicked(){ self.analyse(); }
                    if ui.button("Flip board").clicked(){ self.flip=!self.flip; }
                });
            });
        });
        ctx.request_repaint_after(std::time::Duration::from_millis(70));
    }
}

impl App {
    fn export_pgn(&self, ctx: &egui::Context){
        let fen=self.board.to_string();
        ctx.copy_text(fen.clone());
        self.console.lock().unwrap().push_back(format!("FEN: {}", fen));
    }
    fn handle_click(&mut self, sq: chess::Square, _pos: egui::Pos2, drag: bool){
        if self.thinking || self.promo_pick.is_some() { return; }
        if let Some(sel)=self.sel {
            if sq==sel { if !drag { self.sel=None; self.legal_targets.clear(); } return; }
            if self.legal_targets.contains(&sq){ self.try_move(sel, sq); return; }
            if let Some(pc)=self.board.piece_on(sq){
                if self.board.color_on(sq)==Some(self.board.side_to_move()){
                    let _=pc; self.sel=Some(sq); self.recompute_legal(sq);
                    if drag { self.drag_from=Some(sq); }
                    return;
                }
            }
            self.sel=None; self.legal_targets.clear();
        } else {
            if self.board.piece_on(sq).is_some() && self.board.color_on(sq)==Some(self.board.side_to_move()){
                self.sel=Some(sq); self.recompute_legal(sq);
                if drag { self.drag_from=Some(sq); self.drag_pos=Some(_pos); }
            }
        }
    }
    fn handle_drop(&mut self, sq: chess::Square){
        let from=self.drag_from.take(); self.drag_pos=None;
        if let Some(f)=from{ if f!=sq && self.legal_targets.contains(&sq){ self.try_move(f, sq); } }
    }
    fn recompute_legal(&mut self, from: chess::Square){
        self.legal_targets.clear();
        for mv in chess::MoveGen::new_legal(&self.board){ if mv.get_source()==from { self.legal_targets.push(mv.get_dest()); } }
    }
    fn try_move(&mut self, from: chess::Square, to: chess::Square){
        let pc=self.board.piece_on(from);
        let is_pawn = pc==Some(chess::Piece::Pawn);
        let need_promo = is_pawn && (to.get_rank()==chess::Rank::Eighth || to.get_rank()==chess::Rank::First);
        if need_promo { self.promo_pick=Some((from,to)); return; }
        let mv=chess::ChessMove::new(from,to,None);
        if !self.board.legal(mv){ return; }
        self.push_human(mv);
        self.sel=None; self.legal_targets.clear();
    }
    fn push_human(&mut self, mv: chess::ChessMove){
        let san = format!("{}{}", mv.get_source(), mv.get_dest());
        self.history.push(self.board); self.half_moves.push(san);
        self.board = self.board.make_move_new(mv); self.last_move=Some(mv);
        if self.board.status()!=chess::BoardStatus::Ongoing {
            self.status=format!("Game over: {:?}", self.board.status()); return;
        }
        let m=self.mode.clone();
        if m=="Human vs Human" { return; }
        if m=="Analysis" { self.analyse(); return; }
        let want = if m=="Human vs Owen" { "owen" } else if m=="Human vs Opponent" { "opp" } else { "owen" };
        self.engine_go(want);
    }
    fn new_game(&mut self){
        self.board=chess::Board::default(); self.history.clear(); self.half_moves.clear();
        self.sel=None; self.last_move=None; self.legal_targets.clear();
        self.thinking=false; self.eval_cp=None; self.eval_mate=None; self.pv.clear(); self.promo_pick=None;
        self.start_engines();
        self.status=format!("New game — {}  •  {}", self.mode, self.clock_choice);
        self.console.lock().unwrap().push_back(format!("== New Game {} clock {} Owen={} Opp={}", self.mode, self.clock_choice, self.owen_path, self.opp_path));
        if self.mode=="Owen vs Opponent" { self.engine_go("owen"); }
    }
    fn undo(&mut self){
        if let Some(prev)=self.history.pop(){ self.board=prev; self.half_moves.pop(); self.last_move=None; self.status="Undo.".into(); self.thinking=false; }
    }
    fn start_engines(&mut self){
        self.stop_engines();
        if std::path::Path::new(&self.owen_path).exists(){
            if let Some(mut e)=UciEngine::spawn(&self.owen_path, self.console.clone()){
                if !self.nnue_path.is_empty() && std::path::Path::new(&self.nnue_path).exists(){
                    thread::sleep(std::time::Duration::from_millis(120));
                    e.send(&format!("setoption name NNUEFile value {}", self.nnue_path));
                }
                e.send("isready"); thread::sleep(std::time::Duration::from_millis(40)); e.send("ucinewgame");
                self.owen=Some(e);
            } else { self.console.lock().unwrap().push_back("!! Owen spawn failed".into()); }
        }
        if !self.opp_path.is_empty() && std::path::Path::new(&self.opp_path).exists(){
            if let Some(mut e)=UciEngine::spawn(&self.opp_path, self.console.clone()){
                e.send("isready"); thread::sleep(std::time::Duration::from_millis(40)); e.send("ucinewgame");
                self.opp=Some(e);
            }
        }
    }
    fn stop_engines(&mut self){
        if let Some(mut e)=self.owen.take(){ e.send("stop"); thread::sleep(std::time::Duration::from_millis(20)); e.kill(); }
        if let Some(mut e)=self.opp.take(){ e.send("stop"); thread::sleep(std::time::Duration::from_millis(20)); e.kill(); }
        self.thinking=false;
    }
    fn engine_go(&mut self, which: &str){
        let eng = if which=="owen" { self.owen.as_mut() } else { self.opp.as_mut() };
        let Some(e)=eng else { self.status=format!("No engine for {} — set path", which); return; };
        let fen=self.board.to_string();
        e.send(&format!("position fen {}", fen));
        let go = if self.clock_choice=="no clock" { "go movetime 1200".into() }
            else if self.clock_choice.starts_with("1+") { let t=60_000; format!("go wtime {} btime {} winc 0 binc 0", t, t) }
            else if self.clock_choice.starts_with("3+") { format!("go wtime 180000 btime 180000 winc 2000 binc 2000") }
            else if self.clock_choice.starts_with("10+") { format!("go wtime 600000 btime 600000 winc 5000 binc 5000") }
            else if self.clock_choice.contains("15+") { format!("go wtime 900000 btime 900000 winc 10000 binc 10000") }
            else if self.clock_choice.contains("40/") { format!("go wtime 5400000 btime 5400000 winc 30000 binc 30000") }
            else { format!("go wtime 300000 btime 300000 winc 2000 binc 2000") };
        e.send(&go); self.thinking=true; self.status=format!("{} thinking… {}", which, go);
    }
    fn analyse(&mut self){
        let eng=self.owen.as_mut().or(self.opp.as_mut());
        let Some(e)=eng else { self.status="No engine — set Owen path.".into(); return; };
        e.send(&format!("position fen {}", self.board.to_string()));
        e.send("go infinite"); self.thinking=true; self.status="Analysing… (Stop to end)".into();
    }
}

fn main() -> eframe::Result<()>{
    let opts=eframe::NativeOptions{
        viewport: egui::ViewportBuilder::default().with_inner_size([1180.0, 820.0]).with_min_inner_size([980.0, 680.0]),
        ..Default::default()
    };
    eframe::run_native("Owen 2 — Chess", opts, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
